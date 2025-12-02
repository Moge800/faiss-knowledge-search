import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .faiss_serch import FaissSearch, normalize_katakana_width

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 固有名詞正規化辞書
noun_normalizer = {}


def load_noun_normalizer():
    """noun_base.csvから固有名詞の正規化辞書を作成.

    DATA/noun_base.csvから固有名詞の表記ゆれや略称を読み込み、
    グローバル辞書`noun_normalizer`に格納する。
    CSVには「original」と「normalized」の2列が必要。

    Raises:
        Exception: CSVファイルの読み込みまたはパース時のエラー（警告ログのみ、処理は続行）

    Examples:
        >>> load_noun_normalizer()
        INFO: 固有名詞辞書を読み込みました: 50件
    """
    global noun_normalizer
    try:
        noun_df = pd.read_csv("DATA/noun_base.csv")
        noun_normalizer = dict(zip(noun_df["original"], noun_df["normalized"]))
        logger.info(f"固有名詞辞書を読み込みました: {len(noun_normalizer)}件")
    except Exception as e:
        logger.warning(f"固有名詞辞書の読み込みに失敗: {e}")


def normalize_query(text: str) -> str:
    """検索クエリ内の固有名詞を正規化表記に変換.

    `noun_normalizer`辞書を使用して、クエリ内の固有名詞や
    略称を統一された表記に置換する。複数の置換がある場合は順次適用。

    Args:
        text: 検索クエリ文字列

    Returns:
        正規化された検索クエリ文字列

    Examples:
        >>> normalize_query("FastAPIについて")
        "FastAPIについて"  # 辞書にエントリがない場合は元のまま
        >>> normalize_query("FAPIについて")  # 「FAPI」→「FastAPI」の辞書エントリがある場合
        "FastAPIについて"
    """
    normalized_text = text
    for original, normalized in noun_normalizer.items():
        normalized_text = normalized_text.replace(original, normalized)
    return normalized_text


# FAISSインデックスを初期化
faiss_search = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPIアプリケーションのライフサイクルを管理するコンテキストマネージャ.

    起動時に以下の初期化処理を実行:
    - 固有名詞正規化辞書（noun_base.csv）の読み込み
    - FAISSインデックスの構築（knowledge_data.csvから）

    終了時にクリーンアップ処理を実行（現状はログ出力のみ）。

    Args:
        app: FastAPIアプリケーションインスタンス

    Yields:
        None: アプリケーション実行中の制御を返す

    Raises:
        Exception: インデックス構築時のエラー（ログ記録、処理は続行）

    Examples:
        >>> app = FastAPI(lifespan=lifespan)
        INFO: 固有名詞辞書を読み込みました: 50件
        INFO: FAISSインデックスの構築が完了しました
    """
    # アプリケーション起動時の初期化処理
    global faiss_search

    # 固有名詞辞書をロード
    load_noun_normalizer()

    # FAISSインデックスを構築
    try:
        knowledge_path = "DATA/knowledge_data.csv"
        if os.path.exists(knowledge_path):
            faiss_search = FaissSearch(knowledge_path)
            logger.info("FAISSインデックスの構築が完了しました")
        else:
            logger.error(f"知識データファイルが見つかりません: {knowledge_path}")
    except Exception as e:
        logger.error(f"FAISSインデックスの構築に失敗: {e}")

    yield  # アプリケーションの実行

    # アプリケーション終了時のクリーンアップ処理（必要に応じて）
    logger.info("アプリケーションを終了します")


app = FastAPI(
    title="FAISS Knowledge Search API",
    description="LLMのRAGのためのFAISS検索API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """APIのルートエンドポイント - 基本情報とドキュメントを返す.

    利用可能なエンドポイント、バージョン情報、使用例を含む
    APIの概要をJSON形式で返却する。

    Returns:
        Dict[str, Any]: API情報を含む辞書
            - message (str): APIの名称
            - version (str): APIのバージョン
            - endpoints (List[str]): 利用可能なエンドポイント一覧
            - usage (str): 使用方法の説明
            - example (Dict): リクエストパラメータの例

    Examples:
        >>> GET /
        {
            "message": "FAISS Knowledge Search API",
            "version": "1.0.0",
            "endpoints": ["/knowledge/search", "/health"],
            ...
        }
    """
    return {
        "message": "FAISS Knowledge Search API",
        "version": "1.0.0",
        "endpoints": ["/knowledge/search", "/health"],
        "usage": "POST /knowledge/search with parameters: text (str), top_k (int), threshold (float), min_k (int), fallback (bool)",
        "example": {
            "text": "AIについての知識を検索",
            "top_k": 5,
            "threshold": 0.7,
            "min_k": 3,
            "fallback": "true",
        },
    }


@app.post("/knowledge/search")
async def search_knowledge(
    text: str = Query(..., description="検索対象のテキスト"),
    top_k: int = Query(
        3,
        description="返却する上位結果の数（デフォルト3、最大100。ただし実際のデータ件数が上限）",
        ge=1,
        le=100,
    ),
    threshold: float = Query(
        0.5,
        description="類似度スコアの閾値（デフォルト0.5、0.0〜1.0の範囲）",
        ge=0.0,
        le=1.0,
    ),
    min_k: int = Query(
        3, description="閾値未満の場合に再検索する最小件数（デフォルト3）", ge=1, le=100
    ),
    fallback: bool = Query(False, description="閾値未満の場合に再検索を行うかどうか"),
) -> Dict[str, Any]:
    """知識ベースから意味的に類似したコンテンツをFAISS検索で取得.

    埋め込みモデル（intfloat/multilingual-e5-large）を用いた意味ベースの
    類似度検索を実行。固有名詞正規化とカタカナ正規化を適用し、
    閾値によるフィルタリングとフォールバック機能をサポート。

    Args:
        text: 検索対象のテキスト（必須）。空文字列や空白のみの場合は400エラー
        top_k: 返却する上位結果の数。デフォルト3、範囲1〜100。
            データ総件数を超える場合は全件返却
        threshold: 類似度スコアの閾値（コサイン類似度）。デフォルト0.5、範囲0.0〜1.0。
            この値以上のスコアを持つ結果のみ返却
        min_k: fallback=Trueの場合に再検索する最小件数。デフォルト3、範囲1〜100。
            結果数がmin_k未満の場合、閾値を半減して再検索を繰り返す
        fallback: 閾値未満の結果が少ない場合に再検索を行うかどうか。
            デフォルトFalse。Trueの場合はmin_k件確保するまで再検索

    Returns:
        Dict[str, Any]: 検索結果を含むJSON
            - query (str): 元の検索クエリ
            - normalized_query (str): 正規化後の検索クエリ
            - requested_count (int): リクエストされた件数
            - total_data_count (int): データベース内の総件数
            - actual_returned_count (int): 実際に返却された件数
            - results (List[Dict]): 検索結果のリスト。各要素は以下を含む:
                - rank (int): ランキング順位
                - similarity_score (float): 類似度スコア（0.0〜1.0）
                - その他CSVの全カラム

    Raises:
        HTTPException:
            - 400: 検索テキストが空の場合
            - 500: FAISSインデックスが未初期化、または検索処理エラー

    Examples:
        >>> POST /knowledge/search?text=FastAPI&top_k=3&threshold=0.5
        {
            "query": "FastAPI",
            "normalized_query": "FastAPI",
            "requested_count": 3,
            "total_data_count": 100,
            "actual_returned_count": 3,
            "results": [
                {
                    "rank": 1,
                    "similarity_score": 0.87,
                    "title": "FastAPIとは",
                    "content": "..."
                },
                ...
            ]
        }

    Notes:
        - E5モデルの特性上、クエリには"query: "プレフィックスが自動付与される
        - fallback=Trueの場合、閾値が0.01未満になると再検索を中止
        - 正規化処理: 半角カタカナ→全角カタカナ、固有名詞の統一表記化
    """

    if faiss_search is None:
        raise HTTPException(status_code=500, detail="FAISSインデックスが初期化されていません")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="検索テキストが空です")

    try:
        # クエリを正規化
        temp = normalize_katakana_width(text)
        normalized_query = normalize_query(temp.strip())
        if text != normalized_query:
            logger.info(f"検索クエリ: '{text}' -> 正規化後: '{normalized_query}'")
        else:
            logger.info(f"検索クエリ: '{text}'")

        # データベース内の総件数を取得
        total_data_count = len(faiss_search.data)

        # FAISS検索実行（データ件数以上は要求できない）
        actual_n = min(top_k, total_data_count)
        if fallback:
            results = faiss_search.search_with_fallback(
                normalized_query, actual_n, threshold, min_k
            )
        else:
            results = faiss_search.search(normalized_query, actual_n, threshold)

        response = {
            "query": text,
            "normalized_query": normalized_query,
            "requested_count": top_k,
            "total_data_count": total_data_count,
            "actual_returned_count": len(results),
            "results": results,
        }

        if top_k > total_data_count:
            logger.info(
                f"検索完了: 要求件数{top_k}件に対してデータベース内の総件数{total_data_count}件のため、{len(results)}件を返却"
            )
        else:
            logger.info(f"検索完了: {len(results)}件の結果を返却")

        return response

    except Exception as e:
        logger.error(f"検索エラー: {e}")
        raise HTTPException(status_code=500, detail=f"検索処理でエラーが発生しました: {str(e)}")


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント - システムの状態を返す.

    APIサーバーの稼働状況、FAISSインデックスの準備状況、
    固有名詞辞書のロード状況、データ総件数、使用モデル名を確認。
    ロードバランサーや監視システムからの定期的なヘルスチェックに使用。

    Returns:
        Dict[str, Any]: システム状態を含むJSON
            - status (str): 全体のステータス（"healthy"）
            - faiss_ready (bool): FAISSインデックスが初期化済みか
            - noun_normalizer_loaded (bool): 固有名詞辞書がロード済みか（件数>0）
            - total_data_count (int): インデックス化されたデータ総件数
            - model_name (str | None): 使用中の埋め込みモデル名

    Examples:
        >>> GET /health
        {
            "status": "healthy",
            "faiss_ready": true,
            "noun_normalizer_loaded": true,
            "total_data_count": 100,
            "model_name": "intfloat/multilingual-e5-large"
        }

    Notes:
        - FAISSインデックスが未初期化でも200を返す（faiss_ready=false）
        - 異常系の判定は呼び出し側で実施すること
    """
    return {
        "status": "healthy",
        "faiss_ready": faiss_search is not None,
        "noun_normalizer_loaded": len(noun_normalizer) > 0,
        "total_data_count": len(faiss_search.data) if faiss_search else 0,
        "model_name": faiss_search.model_name if faiss_search else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
