#!/usr/bin/env python3
"""
異なるカラム構造のCSVファイルでのFAISS検索テスト
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.faiss_serch import FaissSearch


def test_different_csv_structures():
    """異なるCSV構造でのテスト"""

    print("=== 異なるCSV構造でのFAISS検索テスト ===\n")

    csv_files = [
        ("DATA/knowledge_data.csv", "標準構造 (id, title, content, category)"),
        (
            "DATA/test_knowledge.csv",
            "拡張構造 (id, name, description, url, author, tags)",
        ),
    ]

    for csv_path, description in csv_files:
        if not os.path.exists(csv_path):
            print(f"❌ ファイルが見つかりません: {csv_path}")
            continue

        print(f"📂 テスト対象: {description}")
        print(f"   ファイル: {csv_path}")

        try:
            # FAISSインデックスを構築
            faiss_search = FaissSearch(csv_path)
            print("✅ インデックス構築成功")
            print(f"   データ件数: {len(faiss_search.data)}")
            print(f"   使用カラム: {faiss_search.text_columns}")
            print(f"   全カラム: {list(faiss_search.data.columns)}")

            # 検索テスト
            test_queries = ["Python", "API", "機械学習"]

            for query in test_queries:
                print(f"\n🔍 検索クエリ: '{query}'")
                results = faiss_search.search(query, 2)

                if results:
                    for result in results:
                        print(
                            f"  ランク {result['rank']}: スコア {result['similarity_score']:.4f}"
                        )
                        # 動的にカラムを表示
                        for key, value in result.items():
                            if key not in ["rank", "similarity_score"]:
                                # 長い文字列は省略
                                if isinstance(value, str) and len(value) > 50:
                                    value = value[:50] + "..."
                                print(f"    {key}: {value}")
                        print()
                else:
                    print("  結果なし")

        except Exception as e:
            print(f"❌ エラー: {e}")

        print("\n" + "=" * 60 + "\n")


def test_missing_columns():
    """必須カラムが欠けた場合のテスト"""

    print("=== 最小構成CSVでのテスト ===")

    # 最小構成のテストCSVを作成
    minimal_csv_content = """item,info
製品A,高品質な製品です
製品B,コストパフォーマンスに優れています
製品C,最新技術を使用しています"""

    minimal_csv_path = "DATA/minimal_test.csv"

    try:
        with open(minimal_csv_path, "w", encoding="utf-8") as f:
            f.write(minimal_csv_content)

        print(f"📂 最小構成ファイル作成: {minimal_csv_path}")

        faiss_search = FaissSearch(minimal_csv_path)
        print("✅ 最小構成でもインデックス構築成功")
        print(f"   使用カラム: {faiss_search.text_columns}")

        # 検索テスト
        results = faiss_search.search("製品", 2)
        print(f"🔍 検索結果: {len(results)}件")

        for result in results:
            print(f"  ランク {result['rank']}: スコア {result['similarity_score']:.4f}")
            for key, value in result.items():
                if key not in ["rank", "similarity_score"]:
                    print(f"    {key}: {value}")
            print()

        # テストファイルを削除
        os.remove(minimal_csv_path)
        print(f"🗑️  テストファイル削除: {minimal_csv_path}")

    except Exception as e:
        print(f"❌ エラー: {e}")
        if os.path.exists(minimal_csv_path):
            os.remove(minimal_csv_path)


if __name__ == "__main__":
    test_different_csv_structures()
    test_missing_columns()
