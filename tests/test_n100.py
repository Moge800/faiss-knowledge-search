import json
import time

import requests


def test_n100():
    """top_k=100を指定した検索テスト"""

    base_url = "http://localhost:8000"

    print("=== top_k=100 検索テスト ===")

    # サーバーが起動するまで少し待つ
    print("サーバーの起動を待機中...")
    time.sleep(3)

    # ヘルスチェック
    print("1. ヘルスチェック")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"ステータス: {response.status_code}")
        health_data = response.json()
        print(f"レスポンス: {json.dumps(health_data, ensure_ascii=False, indent=2)}")

        if "total_data_count" in health_data:
            print(f"データベース内の総件数: {health_data['total_data_count']}件")
    except Exception as e:
        print(f"エラー: {e}")
        return

    print("\n" + "=" * 50 + "\n")

    # top_k=100での検索テスト
    print("2. top_k=100 検索テスト")
    try:
        query = "FAISS"
        response = requests.post(
            f"{base_url}/knowledge/search",
            params={"text": query, "top_k": 100},
            timeout=10,
        )
        print(f"ステータス: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"クエリ: {data['query']}")
            print(f"正規化後: {data.get('normalized_query', 'N/A')}")
            print(f"要求件数: {data.get('requested_count', 'N/A')}")
            print(f"データベース総件数: {data.get('total_data_count', 'N/A')}")
            print(
                f"実際の返却件数: {data.get('actual_returned_count', len(data.get('results', [])))}"
            )

            print("\n検索結果:")
            results = data.get("results", [])
            for result in results:
                print(f"  ランク{result['rank']}: {result['title']}")
                print(f"    類似度: {result['similarity_score']:.4f}")
                print(f"    内容: {result['content'][:80]}...")
                print()
        else:
            print(f"エラーレスポンス: {response.text}")

    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    test_n100()
