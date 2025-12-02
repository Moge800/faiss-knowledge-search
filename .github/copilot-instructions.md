# GitHub Copilot Instructions

## プロジェクト概要
FAISS（Facebook AI Similarity Search）を使用した高性能知識検索APIシステム。LLMのRAG（Retrieval-Augmented Generation）向けに設計され、FastAPIベースのRESTful APIとして実装。多言語対応の埋め込みモデル（intfloat/multilingual-e5-large）を用いた意味ベースの類似度検索を提供。

## 技術スタック
- **Python**: 3.8+
- **パッケージマネージャ**: uv
- **APIフレームワーク**: FastAPI 0.117.1, Uvicorn
- **検索エンジン**: FAISS (faiss-cpu 1.7.4+)
- **埋め込みモデル**: sentence-transformers 2.2.0+ (intfloat/multilingual-e5-large)
- **データ処理**: pandas 2.0.0+
- **日本語正規化**: jaconv 0.3.4+
- **データ検証**: Pydantic
- **開発ツール**: pytest, pytest-asyncio, black, isort

## プロジェクト構造
```
.
├── src/
│   ├── __init__.py
│   ├── main_app.py           # FastAPIアプリケーション本体
│   └── faiss_serch.py        # FAISS検索エンジンクラス
├── DATA/
│   ├── knowledge_data.csv    # 知識データベース（本番用）
│   ├── noun_base.csv         # 固有名詞正規化辞書
│   └── test_knowledge.csv    # テスト用データ
├── tests/
│   ├── test_api.py           # APIエンドポイントテスト
│   ├── test_n100.py          # 大量検索テスト
│   └── test_dynamic_columns.py # 動的カラム対応テスト
├── docs/
│   ├── advanced_search.md    # 高度な検索機能の説明
│   ├── model_cache.md        # モデルキャッシュの仕組み
│   └── dynamic_columns.md    # 動的カラム対応の詳細
├── start_server.py           # サーバー起動スクリプト
├── pyproject.toml            # プロジェクト設定（主）
└── requirements.txt          # 依存関係リスト
```

**モジュール責務分離**:
- `main_app.py`: FastAPIエンドポイント、ライフサイクル管理、固有名詞正規化
- `faiss_serch.py`: FAISSインデックス構築、類似度検索、モデル管理（シングルトン）

## コーディング規約

### 1. 型ヒントは必須
```python
# Good
def search(self, query_text: str, top_k: int, threshold: float = 0.5) -> List[Dict[str, Any]]:
    return results

# Bad
def search(self, query_text, top_k, threshold=0.5):
    return results
```

### 2. 環境変数と設定管理
- データファイルパスは環境変数または設定ファイルで管理
- ハードコーディングを避ける
```python
# Good
KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_DATA_PATH", "DATA/knowledge_data.csv")

# Bad
knowledge_path = "DATA/なれっじ.csv"
```

### 3. エラーハンドリング
- `Exception`の汎用捕捉は避ける
- 具体的な例外を指定: `FileNotFoundError`, `ValueError`, `HTTPException`など
```python
# Good
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError as e:
    logger.error(f"CSVファイルが見つかりません: {e}")
    raise

# Bad
except Exception as e:
    pass
```

### 4. マジックナンバーは定数化
```python
# Good
DEFAULT_TOP_K = 3
MAX_TOP_K = 100
DEFAULT_THRESHOLD = 0.5

# Bad
top_k: int = Query(3, ..., ge=1, le=100)
```

### 5. グローバル変数は最小限に
- シングルトンパターンを活用（`ModelManager`クラス）
- アプリケーションレベルの状態は`lifespan`で管理

### 6. インポート順序
```python
# 標準ライブラリ
import os
import logging
from typing import Dict, Any, List
from contextlib import asynccontextmanager

# サードパーティ
import faiss
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sentence_transformers import SentenceTransformer

# ローカル
from .faiss_serch import FaissSearch, normalize_katakana_width
```

### 7. ロギング
- 共通ロガーを使用
- ログレベル: DEBUG, INFO, WARNING, ERROR
```python
import logging

logger = logging.getLogger(__name__)
logger.info("FAISSインデックスの構築が完了しました")
logger.error(f"検索エラー: {e}")
```

### 8. FAISSとsentence-transformersの使用パターン
```python
# モデル初期化（シングルトン）
model = SentenceTransformer("intfloat/multilingual-e5-large")

# E5モデルのプレフィックス付与
texts = [f"passage: {text}" for text in texts]
query_text = f"query: {query_text}"

# ベクトル正規化（コサイン類似度用）
vectors = model.encode(texts, normalize_embeddings=True)

# FAISS IndexFlatIP（内積検索）
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors.astype("float32"))
```

### 9. 非同期処理（FastAPI）
```python
# Good
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時処理
    yield
    # 終了時処理

# エンドポイントは必要に応じて非同期化
@app.post("/knowledge/search")
async def search_knowledge(...) -> Dict[str, Any]:
    ...
```

### 10. Pydanticによるデータ検証
```python
# Good（将来的な改善）
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, description="検索対象のテキスト")
    top_k: int = Field(3, ge=1, le=100)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
```

## FAISS検索の注意点
- **IndexFlatIP**: 内積検索（正規化ベクトルでコサイン類似度に相当）
- **モデルキャッシュ**: `ModelManager`シングルトンで共有
- **動的カラム対応**: CSV構造に応じて自動的にテキストカラムを検出
- **カタカナ正規化**: 半角カタカナ→全角カタカナ変換（`jaconv`）
- **固有名詞正規化**: `noun_base.csv`で略称や表記ゆれに対応

## FastAPI特有の考慮事項
- `lifespan`コンテキストマネージャで起動時/終了時処理
- 自動ドキュメント生成（Swagger UI: `/docs`, ReDoc: `/redoc`）
- Queryパラメータのバリデーション（ge, le, description）
- HTTPExceptionでステータスコード指定

## セキュリティ
- `.env`ファイルは`.gitignore`で除外（機密情報を含む場合）
- ログファイルも`.gitignore`で除外
- APIエンドポイントのバリデーション必須

## テスト

### テスト駆動開発(TDD)の推奨
**新機能追加時は必ずテストも同時作成する**

#### テストの配置
```
tests/
├── test_api.py               # APIエンドポイントの統合テスト
├── test_n100.py             # 大量検索のパフォーマンステスト
└── test_dynamic_columns.py   # 動的カラム検出のユニットテスト
```

#### テスト作成ルール
1. **新しいエンドポイント追加** → 対応するテストを`tests/test_api.py`に追加
2. **検索ロジック変更** → `test_api.py`と`test_n100.py`を更新
3. **バグ修正** → 再現テストを追加してから修正

#### FAISSとモデルのテスト
- 実際のモデルダウンロードは必要（初回のみ）
- ダミーデータで動作確認
```python
def test_faiss_search():
    # テスト用CSVを使用
    faiss_search = FaissSearch("DATA/test_knowledge.csv")
    results = faiss_search.search("テストクエリ", top_k=3)
    assert len(results) <= 3
    assert all("similarity_score" in r for r in results)
```

#### APIテストの注意
- サーバー起動が前提のテスト（`test_api.py`, `test_n100.py`）
- pytest-asyncioで非同期テスト対応（将来的な改善）
```python
# 現在の方式（サーバー起動が必要）
response = requests.post("http://localhost:8000/knowledge/search", params={"text": query, "top_k": 3})

# 推奨方式（TestClientを使用）
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.post("/knowledge/search", params={"text": query, "top_k": 3})
```

#### テスト実行コマンド
```bash
# 全テスト実行
pytest tests/ -v

# 特定のテストファイルのみ
pytest tests/test_api.py -v

# カバレッジ計測
pytest --cov=src tests/
```

#### テストの命名規則
- ファイル: `test_*.py`
- 関数: `test_*` (例: `test_search_basic`, `test_fallback_mechanism`)

## デプロイ
- `uv sync`で依存関係インストール
- `python start_server.py`または`uvicorn src.main_app:app --host 0.0.0.0 --port 8000`で起動
- ヘルスチェック: `GET /health`

## よくある問題と解決策

### データファイルが見つからない
- `start_server.py`でファイル存在チェック実装済み
- パスの一貫性を保つ（`knowledge_data.csv`と`なれっじ.csv`の混在を解消）

### モデルダウンロードに時間がかかる
- 初回のみ約2.5GB（intfloat/multilingual-e5-large）
- キャッシュ場所: `~/.cache/huggingface/`（Linux/macOS）、`C:\Users\<user>\.cache\huggingface\`（Windows）

### カラム構造が異なるCSV
- 自動検出機能により対応可能
- ログで使用カラムを確認: `INFO:src.faiss_serch:検出されたテキストカラム: ['title', 'content']`

### APIエンドポイントの不一致
- **正**: `/knowledge/search` (POST)
- README.mdとの整合性を保つ

## コード品質
- Black: フォーマッター(自動整形)
- isort: import順序整理
- pytest: テストフレームワーク
- mypy: 型チェック（`pyproject.toml`で設定済み）

## 命名規則
- クラス: `PascalCase` (例: `FaissSearch`, `ModelManager`, `IndexData`)
- 関数/変数: `snake_case` (例: `search_knowledge`, `normalize_query`, `top_k`)
- 定数: `UPPER_SNAKE_CASE` (例: `DEFAULT_TOP_K`, `MAX_TOP_K`)
- プライベート: `_leading_underscore` (例: `_instance`, `_model`)

## ドキュメント
- Docstring: Google Style推奨
- 型ヒントで大部分は自己文書化
- 複雑な検索ロジックにはインラインコメント
- `docs/`フォルダに詳細ドキュメント配置

## 定期メンテナンス手順

### 大きな変更時・仕事終わりのチェックリスト

#### 1. 全スキャンによるテスト項目チェック
大きな機能追加や1日の開発終了時に、テストの過不足をチェック:

```bash
# 全Pythonファイルをスキャン
semantic_search "FAISS search API endpoints business logic"

# 既存テストと比較して未カバーのコンポーネントを特定
```

**チェック対象**:
- [ ] 新規追加した検索機能にテストがあるか
- [ ] 修正した類似度計算ロジックのテストケースが十分か
- [ ] 新しいエンドポイントに統合テストがあるか
- [ ] エラーハンドリング（404, 500など）のテストが網羅されているか
- [ ] 動的カラム検出の各種パターンがテストされているか

**テスト追加が必要な場合**:
1. 該当コンポーネントのテストファイルを作成/更新 (`tests/test_<component>.py`)
2. `pytest tests/ -v` で全テスト実行し、合格を確認
3. GitHub Actionsでも自動テストが通ることを確認（設定されている場合）

#### 2. 開発ログの作成
その日の開発内容をまとめた資料を`dev_logs/`フォルダに生成:

```bash
# ファイル名: dev_logs/YYYY-MM-DD.md (例: dev_logs/2025-12-03.md)
```

**ログに記載する内容**:
- **実施内容サマリー**: 何を追加・修正したか
- **テスト結果**: 新規テスト数、全体のテスト実行結果
- **カバレッジ分析**: テスト済みモジュールと未テスト箇所
- **発見した課題と対応**: バグや改善点（例: ファイルパス不整合、エンドポイント名の統一）
- **CI/CD確認**: GitHub Actionsの結果（設定されている場合）
- **今後の改善案**: 次回以降のTODO

**テンプレート構成例**:
```markdown
# 開発ログ - YYYY-MM-DD

## 📋 実施内容サマリー
...

## 📝 詳細レポート
### A. 新規機能/修正
...

## 🧪 テスト実行結果
...

## 📊 カバレッジ分析
...

## 🔍 発見した課題と対応
...

## ✅ CI/CD確認
...

## 🚀 今後の改善案
...

## 📌 まとめ
...
```

#### 3. コードフォーマット・Lint確認
```bash
# 1. フォーマット適用
black src/ tests/
isort src/ tests/

# 2. 型チェック
mypy src/

# 3. 全テスト実行
pytest tests/ -v --tb=short

# 4. 変更差分確認
git status
git diff

# 5. コミット
git add .
git commit -m "feat: <変更内容の要約>"
git push origin main
```

#### 4. API仕様の整合性確認
```bash
# Swagger UIで確認
# http://localhost:8000/docs

# ヘルスチェック
curl http://localhost:8000/health

# 基本検索テスト
curl -X POST "http://localhost:8000/knowledge/search?text=FastAPI&top_k=3"
```

### 推奨頻度
- **テストチェック**: 大きな機能追加時 or 1日の終わり
- **開発ログ作成**: 1日の終わり (複数日にまたがる場合は節目で)
- **フォーマット・Lint**: 毎コミット前
- **カバレッジ計測**: 週1回 or リリース前

### メリット
- テストの抜け漏れを早期発見
- 開発履歴が明確に記録される
- API仕様とドキュメントの整合性維持
- 将来の自分が過去の判断を理解できる

## 既知の改善課題（2025-12-03時点）

### 優先度: 高
1. **ファイルパスの統一**: `knowledge_data.csv` vs `なれっじ.csv` の不整合解消
2. **APIエンドポイント名の統一**: `/knowledge/search` vs `/get_knowledge` の整合性確保
3. **テストパラメータ名の修正**: `n` → `top_k` への統一

### 優先度: 中
4. **ファイル名のタイポ修正**: `faiss_serch.py` → `faiss_search.py`
5. **requirements.txtの簡素化**: 直接依存のみを記載、`pyproject.toml`を主とする
6. **型ヒントの完全化**: 全関数に戻り値の型ヒントを追加

### 優先度: 低
7. **ヘルスチェックステータス値**: `"HAPPY"` → `"healthy"` へ変更
8. **TestClient導入**: サーバー起動不要なテスト環境の整備
9. **Pydanticモデル導入**: リクエスト/レスポンスの型安全性向上

---

**このプロジェクトはRAG実装の学習・実験目的で開発されています。質問や改善提案は歓迎します！**
