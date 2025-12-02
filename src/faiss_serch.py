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
    if isinstance(text, str):
        return re.sub(
            r"[ｦ-ﾟ]+",
            lambda m: jaconv.h2z(m.group(), kana=True, ascii=False, digit=False),
            text,
        )
    return text


@dataclass
class IndexData:
    data: pd.DataFrame
    index: faiss.IndexFlatIP
    text_columns: List[str]


class ModelManager:
    """モデルのシングルトン管理クラス"""

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
        """モデルをキャッシュから取得、初回のみダウンロード"""
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
        """テキストカラムを自動検出"""
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

            vectors = self.model.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            )
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors.astype("float32"))

            logger.info(
                f"FAISSインデックス作成完了: {index.ntotal}件, 次元数: {vectors.shape[1]}"
            )

            return IndexData(data=data, index=index, text_columns=text_columns)

        except Exception as e:
            logger.error(f"make_index失敗: {e}")
            raise e

    def search(
        self, query_text: str, top_k: int, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
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
        """類似度スコアの閾値を用いた検索。閾値以下の場合はmin_kになるまでしきい値を下げ繰り返し再検索"""
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
