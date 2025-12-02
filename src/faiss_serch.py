import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import faiss
import jaconv
import pandas as pd
from sentence_transformers import SentenceTransformer

# ロガーの設定
logger = logging.getLogger(__name__)


def normalize_katakana_width(text: str) -> str:
    """半角カタカナを全角カタカナに変換.

    jaconvライブラリを使用して、テキスト内の半角カタカナ（ｦ-ﾟ）を
    全角カタカナに統一する。検索時の表記ゆれを吸収するために使用。

    Args:
        text: 変換対象のテキスト

    Returns:
        全角カタカナに変換されたテキスト

    Examples:
        >>> normalize_katakana_width("ﾃｽﾄ")
        "テスト"
        >>> normalize_katakana_width("テスト")
        "テスト"
        >>> normalize_katakana_width("Test ﾃｽﾄ")
        "Test テスト"

    Notes:
        - ASCII文字と数字は変換対象外
        - 文字列以外が渡された場合はそのまま返却
    """
    if isinstance(text, str):
        return re.sub(
            r"[ｦ-ﾟ]+",
            lambda m: jaconv.h2z(m.group(), kana=True, ascii=False, digit=False),
            text,
        )
    return text


@dataclass
class IndexData:
    """FAISSインデックスとメタデータを保持するデータクラス.

    Attributes:
        data: 元のCSVデータを格納したDataFrame
        index: FAISS IndexFlatIP（内積検索インデックス）
        text_columns: 埋め込みベクトル生成に使用したカラム名のリスト

    Examples:
        >>> index_data = IndexData(
        ...     data=pd.DataFrame({"title": ["A"], "content": ["B"]}),
        ...     index=faiss.IndexFlatIP(768),
        ...     text_columns=["title", "content"]
        ... )
    """

    data: pd.DataFrame
    index: faiss.IndexFlatIP
    text_columns: List[str]


class ModelManager:
    """埋め込みモデルのシングルトン管理クラス.

    sentence-transformersモデルを一度だけロードし、複数のFaissSearchインスタンス間で
    共有することでメモリ使用量を削減し、起動時間を短縮する。

    Attributes:
        _instance: シングルトンインスタンス（クラス変数）
        _model: キャッシュされたSentenceTransformerモデル（クラス変数）
        model_name: 使用中のモデル名（インスタンス変数）

    Examples:
        >>> manager1 = ModelManager()
        >>> manager2 = ModelManager()
        >>> manager1 is manager2
        True
        >>> model = manager1.get_model("intfloat/multilingual-e5-large")
        INFO: モデルを初期化中: intfloat/multilingual-e5-large
        INFO: モデル初期化完了

    Notes:
        - __new__メソッドでシングルトンパターンを実装
        - モデルは初回呼び出し時にのみダウンロード・キャッシュされる
        - デフォルトモデル: intfloat/multilingual-e5-large (約2.5GB)
    """

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # "paraphrase-multilingual-MiniLM-L12-v2"
    # "intfloat/multilingual-e5-large"
    #  "intfloat/e5-base-v2"
    #  "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
    def get_model(self, model_name: str = "intfloat/multilingual-e5-large"):
        """埋め込みモデルをキャッシュから取得、初回のみダウンロード.

        既にモデルがキャッシュされている場合はそれを返却。
        初回呼び出し時はHugging Face Hubからダウンロードしてキャッシュ。

        Args:
            model_name: 使用するモデル名。デフォルトは"intfloat/multilingual-e5-large"。
                他の選択肢:
                - "paraphrase-multilingual-MiniLM-L12-v2" (軽量)
                - "intfloat/e5-base-v2" (英語特化)
                - "sonoisa/sentence-bert-base-ja-mean-tokens-v2" (日本語特化)

        Returns:
            SentenceTransformer: キャッシュされた埋め込みモデル

        Raises:
            Exception: モデルのダウンロードまたは初期化時のエラー

        Examples:
            >>> manager = ModelManager()
            >>> model = manager.get_model()
            INFO: モデルを初期化中: intfloat/multilingual-e5-large
            INFO: モデル初期化完了
            >>> model.encode(["テスト"])
            array([[-0.01, 0.02, ...]], dtype=float32)

        Notes:
            - キャッシュ場所: ~/.cache/huggingface/ (Linux/macOS)
            - キャッシュ場所: C:\\Users\\<user>\\.cache\\huggingface\\ (Windows)
            - 初回ダウンロードには数分〜数十分かかる場合がある
        """
        self.model_name = model_name
        if self._model is None:
            logger.info(f"モデルを初期化中: {model_name}")
            try:
                self._model = SentenceTransformer(model_name)
            except Exception as e:
                logger.error(f"モデルの初期化に失敗しました: {e}")
                raise e
            logger.info("モデル初期化完了")
        return self._model


class FaissSearch:
    """FAISS（Facebook AI Similarity Search）を用いた高速類似度検索エンジン.

    CSVファイルから知識データを読み込み、埋め込みモデルでベクトル化し、
    FAISSインデックスを構築。意味的に類似したコンテンツを高速に検索する。

    Attributes:
        model_manager: シングルトンのModelManagerインスタンス
        model: 使用中のSentenceTransformerモデル
        model_name: モデル名（例: "intfloat/multilingual-e5-large"）
        index_data: IndexDataインスタンス（data, index, text_columns）
        data: 元のCSVデータ（DataFrame）
        index: FAISSインデックス（IndexFlatIP）
        text_columns: ベクトル化に使用したカラム名のリスト

    Examples:
        >>> faiss_search = FaissSearch("DATA/knowledge_data.csv")
        INFO: モデルを初期化中: intfloat/multilingual-e5-large
        INFO: モデル初期化完了
        INFO: 検出されたテキストカラム: ['title', 'content']
        INFO: FAISSインデックス作成完了: 100件, 次元数: 1024

        >>> results = faiss_search.search("FastAPI", top_k=3, threshold=0.5)
        >>> len(results)
        3
        >>> results[0]["similarity_score"]
        0.87

    Notes:
        - IndexFlatIP: 内積検索（正規化ベクトルでコサイン類似度に相当）
        - E5モデル使用時は"query: "と"passage: "プレフィックスを自動付与
        - 動的カラム検出により任意のCSV構造に対応
    """

    def __init__(self, csv_path: str):
        # シングルトンのモデルマネージャーを使用
        self.model_manager = ModelManager()
        self.model = self.model_manager.get_model()
        self.model_name = self.model_manager.model_name
        self.index_data: IndexData = self.make_index(csv_path)
        self.data = self.index_data.data
        self.index = self.index_data.index
        self.text_columns = self.index_data.text_columns

    def detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """DataFrameからテキスト化可能なカラムを自動検出.

        優先順位に基づいてテキストカラムを検出:
        1. 優先カラム名（title, content, name, description, text, summary, body）
        2. 文字列型（object）で除外リスト外のカラム
        3. 上記が見つからない場合、'id'以外の最初のカラム

        Args:
            df: 検査対象のDataFrame

        Returns:
            検出されたテキストカラム名のリスト（最低1つ）

        Raises:
            ValueError: テキスト化可能なカラムが1つも見つからない場合

        Examples:
            >>> df = pd.DataFrame({"id": [1], "title": ["A"], "content": ["B"]})
            >>> faiss_search.detect_text_columns(df)
            ['title', 'content']

            >>> df = pd.DataFrame({"id": [1], "description": ["C"]})
            >>> faiss_search.detect_text_columns(df)
            ['description']

        Notes:
            - 除外カラム: id, category, tag, url, author（メタデータとして扱う）
            - 優先カラムが複数ある場合は全て使用（スペース区切りで結合）
        """
        text_columns = []

        # 優先順位でテキストカラムを検出
        priority_columns = [
            "title",
            "content",
            "name",
            "description",
            "text",
            "summary",
            "body",
        ]

        for col in priority_columns:
            if col in df.columns:
                text_columns.append(col)

        # 優先カラムが見つからない場合、文字列型のカラムを自動検出
        if not text_columns:
            for col in df.columns:
                if df[col].dtype == "object" and col.lower() not in [
                    "id",
                    "category",
                    "tag",
                    "url",
                    "author",
                ]:
                    text_columns.append(col)

        # 最低限1つのテキストカラムが必要
        if not text_columns:
            # 'id'以外の最初のカラムを使用
            available_cols = [col for col in df.columns if col.lower() != "id"]
            if available_cols:
                text_columns = [available_cols[0]]
            else:
                raise ValueError("テキスト化可能なカラムが見つかりません")

        return text_columns

    def make_index(self, csv_path: str) -> IndexData:
        """CSVファイルからFAISSインデックスを構築.

        処理フロー:
        1. CSVファイルを読み込み
        2. テキストカラムを自動検出
        3. 各行のテキストカラムを結合してベクトル化
        4. FAISSインデックス（IndexFlatIP）を作成

        Args:
            csv_path: 知識データのCSVファイルパス

        Returns:
            IndexData: data（DataFrame）、index（FAISS）、text_columnsを含むデータクラス

        Raises:
            Exception: CSVの読み込み、ベクトル化、インデックス構築時のエラー

        Examples:
            >>> index_data = faiss_search.make_index("DATA/knowledge_data.csv")
            INFO: 検出されたテキストカラム: ['title', 'content']
            INFO: FAISSインデックス作成完了: 100件, 次元数: 1024
            >>> index_data.index.ntotal
            100
            >>> index_data.data.shape
            (100, 5)

        Notes:
            - E5モデル使用時は各テキストに"passage: "プレフィックスを付与
            - ベクトルはL2正規化（normalize_embeddings=True）
            - IndexFlatIPは正規化ベクトルの内積でコサイン類似度を計算
            - NaN値は自動的にスキップされる
        """
        try:
            data = pd.read_csv(csv_path)
            text_columns = self.detect_text_columns(data)
            logger.info(f"検出されたテキストカラム: {text_columns}")

            def preprocess_row(row):
                return " ".join(
                    [
                        normalize_katakana_width(str(row[col]))
                        for col in text_columns
                        if pd.notna(row[col])
                    ]
                )

            texts = [preprocess_row(row) for _, row in data.iterrows()]

            if "e5" in self.model_name:
                texts = [f"passage: {t}" for t in texts]

            vectors = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors.astype("float32"))

            logger.info(f"FAISSインデックス作成完了: {index.ntotal}件, 次元数: {vectors.shape[1]}")

            return IndexData(data=data, index=index, text_columns=text_columns)

        except Exception as e:
            logger.error(f"make_index失敗: {e}")
            raise e

    def search(self, query_text: str, top_k: int, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """クエリに意味的に類似した上位k件を検索.

        クエリテキストをベクトル化し、FAISSインデックスで類似度検索を実行。
        閾値以上のスコアを持つ結果のみを返却。

        Args:
            query_text: 検索クエリ文字列
            top_k: 返却する上位結果の数（最大値）
            threshold: 類似度スコアの閾値（デフォルト0.5、範囲0.0〜1.0）

        Returns:
            検索結果のリスト（Dict）。各要素は以下を含む:
                - rank (int): ランキング順位（1から開始）
                - similarity_score (float): コサイン類似度スコア
                - その他CSVの全カラム（文字列化、NaNは空文字列）

        Examples:
            >>> results = faiss_search.search("FastAPI", top_k=3, threshold=0.5)
            >>> len(results)
            3
            >>> results[0]
            {
                'rank': 1,
                'similarity_score': 0.87,
                'title': 'FastAPIとは',
                'content': '高速なPython Webフレームワーク',
                ...
            }
            >>> results[0]['similarity_score'] >= 0.5
            True

        Notes:
            - E5モデル使用時はクエリに"query: "プレフィックスを自動付与
            - カタカナは自動的に全角に正規化
            - スコアがthreshold未満の結果は除外される
            - インデックスが-1の結果（該当なし）も除外される
        """
        query_text = normalize_katakana_width(query_text)
        if "e5" in self.model_name:
            query_text = f"query: {query_text}"

        query_vector = self.model.encode(
            [query_text], show_progress_bar=False, normalize_embeddings=True
        )
        distances, indices = self.index.search(query_vector.astype("float32"), top_k)

        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx == -1 or score < threshold:
                continue
            row = self.data.iloc[idx]
            result: Dict[str, Any] = {"rank": i + 1, "similarity_score": float(score)}
            for col in self.data.columns:
                result[col] = str(row[col]) if pd.notna(row[col]) else ""
            results.append(result)

        return results

    def search_with_fallback(
        self, query_text: str, top_k: int = 3, threshold: float = 0.5, min_k: int = 3
    ) -> List[Dict[str, Any]]:
        """閾値による検索を実行し、結果不足時は閾値を下げて再検索.

        初回検索で結果数がmin_k未満の場合、閾値を半減させて再検索を繰り返す。
        最終的にtop_k件に制限して返却。結果が少ない場合でも最低限の候補を確保。

        Args:
            query_text: 検索クエリ文字列
            top_k: 最終的に返却する上位結果の数（デフォルト3）
            threshold: 初期の類似度スコア閾値（デフォルト0.5、範囲0.0〜1.0）
            min_k: 確保したい最小結果数（デフォルト3）

        Returns:
            検索結果のリスト（最大top_k件）。形式はsearch()と同じ。

        Examples:
            >>> # 通常の検索（十分な結果がある場合）
            >>> results = faiss_search.search_with_fallback(
            ...     "FastAPI", top_k=5, threshold=0.7, min_k=3
            ... )
            >>> len(results)
            5

            >>> # フォールバック発動（結果が不足する場合）
            >>> results = faiss_search.search_with_fallback(
            ...     "マイナーなトピック", top_k=5, threshold=0.9, min_k=3
            ... )
            INFO: resultがmin_k[3]に満たないため、thresholdを[0.45]に下げて再検索します。
            >>> len(results)
            3

        Notes:
            - 閾値は毎回半減（threshold / 2）
            - 閾値が0.01未満になると再検索を中止（無意味な低スコアを避ける）
            - 最終結果はtop_k件に切り詰められる（min_k > top_kでも）
            - 再検索時のログレベル: INFO
        """
        results = self.search(query_text, top_k, threshold)

        while len(results) < min_k:
            logger.info(
                f"resultがmin_k[{min_k}]に満たないため、thresholdを[{threshold/2}]に下げて再検索します。"
            )
            threshold = threshold / 2
            results = self.search(query_text, min_k, threshold)
            if len(results) >= min_k:
                break
            if threshold < 0.01:  # あまりに低い閾値は無意味なので打ち切り
                logger.info("閾値が非常に低いため、これ以上の再検索を中止します。")
                break

        # 最終的な結果をtop_k件に制限
        return results[:top_k]
