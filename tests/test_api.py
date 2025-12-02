import json

import requests


def test_api():
    """FAISS Knowledge Search APIのテスト"""

    base_url = "http://localhost:8000"

    print("=== FAISS Knowledge Search API テスト ===\n")

    # 1. ヘルスチェック
    print("1. ヘルスチェック")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"ステータス: {response.status_code}")
        print(f"レスポンス: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"エラー: {e}")

    print("\n" + "=" * 50 + "\n")

    # 2. 基本情報取得
    print("2. API基本情報")
    try:
        response = requests.get(f"{base_url}/")
        print(f"ステータス: {response.status_code}")
        print(f"レスポンス: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"エラー: {e}")

    print("\n" + "=" * 50 + "\n")

    # 3. 検索テスト
    test_queries = [
        ("FastAPI", 3),
        ("機械学習", 2),
        ("RAG", 3),
        ("api", 2),
        ("データ処理", 3),
    ]  # 固有名詞正規化テスト

    for i, (query, top_k) in enumerate(test_queries, 1):
        print(f"3.{i} 検索テスト: '{query}'")
        try:
            response = requests.post(
                f"{base_url}/knowledge/search", params={"text": query, "top_k": top_k}
            )
            print(f"ステータス: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"クエリ: {data['query']}")
                print(f"正規化後: {data['normalized_query']}")
                print(f"結果数: {data['total_results']}")

                for result in data["results"]:
                    print(f"  ランク{result['rank']}: {result['title']}")
                    print(f"    類似度: {result['similarity_score']:.4f}")
                    print(f"    内容: {result['content'][:50]}...")
                    print(f"    カテゴリ: {result['category']}")
            else:
                print(f"エラーレスポンス: {response.text}")

        except Exception as e:
            print(f"エラー: {e}")

        print("\n" + "-" * 30 + "\n")


if __name__ == "__main__":
    test_api()
