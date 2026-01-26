#!/usr/bin/env python3
"""
多様な日本語コーパス生成
テンプレート + 組み合わせで10000行以上のユニークな文章を生成
"""

import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "japanese_corpus_large.txt"

# 主語
subjects = [
    "人工知能", "深層学習", "機械学習", "スパイキングニューラルネットワーク",
    "ニューロモーフィックコンピューティング", "自然言語処理", "コンピュータビジョン",
    "強化学習", "教師あり学習", "教師なし学習", "転移学習", "メタ学習",
    "ニューラルネットワーク", "畳み込みネットワーク", "再帰型ネットワーク",
    "トランスフォーマー", "BERT", "GPT", "言語モデル", "生成モデル",
    "脳型コンピューティング", "量子コンピューティング", "エッジAI",
    "ロボット工学", "自動運転技術", "音声認識", "画像認識", "物体検出",
    "セマンティックセグメンテーション", "感情分析", "機械翻訳",
    "質問応答システム", "対話システム", "推薦システム", "異常検知",
    "時系列予測", "クラスタリング", "次元削減", "特徴抽出",
    "データマイニング", "知識グラフ", "オントロジー",
    "脳", "ニューロン", "シナプス", "軸索", "樹状突起",
    "海馬", "大脳皮質", "小脳", "視床", "扁桃体",
    "日本語", "英語", "中国語", "言語", "文法", "構文",
    "意味論", "語用論", "形態論", "音韻論",
]

# 述語（動詞フレーズ）
predicates = [
    "急速に進歩している", "発展を続けている", "注目を集めている",
    "実用化が進んでいる", "研究が活発に行われている",
    "新たな可能性を開いている", "革命をもたらしている",
    "効率化を実現している", "自動化を可能にしている",
    "精度を向上させている", "コストを削減している",
    "エネルギー消費を抑えている", "計算時間を短縮している",
    "人間の能力を超えている", "社会に変革をもたらしている",
    "産業に貢献している", "科学の発展に寄与している",
    "新しいアルゴリズムを生み出している", "理論的基盤を提供している",
    "未解決の問題に挑戦している", "画期的な成果を上げている",
    "多くの分野に応用されている", "実世界の問題を解決している",
    "人々の生活を豊かにしている", "未来を変革しようとしている",
]

# 目的語を含む述語
object_predicates = [
    ("効率的な", "処理を可能にする"),
    ("高精度な", "予測を実現する"),
    ("低消費電力の", "計算を実現する"),
    ("リアルタイムの", "分析を可能にする"),
    ("大規模な", "データ処理を行う"),
    ("複雑な", "パターンを認識する"),
    ("抽象的な", "概念を学習する"),
    ("自然な", "言語理解を実現する"),
    ("人間らしい", "応答を生成する"),
    ("創造的な", "コンテンツを生成する"),
    ("正確な", "翻訳を提供する"),
    ("多言語の", "処理を実現する"),
    ("継続的な", "学習を行う"),
    ("適応的な", "振る舞いを示す"),
    ("堅牢な", "システムを構築する"),
]

# 理由・条件フレーズ
reasons = [
    "計算能力の向上により", "大量のデータが利用可能になったことで",
    "新しいアルゴリズムの開発により", "ハードウェアの進歩によって",
    "クラウドコンピューティングの普及で", "オープンソースの発展により",
    "産学連携の進展によって", "国際的な研究協力により",
    "投資の増加によって", "社会的なニーズの高まりから",
    "技術的なブレークスルーにより", "理論的な基盤の確立によって",
    "並列処理技術の発展で", "メモリ効率の改善により",
    "学習データの品質向上によって", "評価手法の確立により",
]

# 接続詞
conjunctions = [
    "そして", "また", "さらに", "一方で", "同時に",
    "加えて", "特に", "例えば", "実際に", "具体的には",
    "結果として", "その結果", "こうして", "したがって",
]

# 修飾語
modifiers = [
    "非常に", "とても", "極めて", "特に", "著しく",
    "飛躍的に", "急速に", "着実に", "確実に", "継続的に",
    "効率的に", "効果的に", "劇的に", "顕著に", "明らかに",
]

# 時間表現
time_expressions = [
    "近年", "最近", "現在", "今日", "21世紀に入って",
    "過去10年で", "ここ数年で", "2020年代に入り",
    "将来的に", "今後", "これから", "やがて",
]

# 場所・分野表現
domains = [
    "医療分野で", "金融業界で", "製造業において", "農業分野で",
    "教育分野で", "エンターテインメント業界で", "交通分野で",
    "セキュリティ分野で", "環境問題において", "エネルギー分野で",
    "通信業界で", "小売業において", "物流分野で", "建設業界で",
    "研究機関で", "大学において", "企業の研究開発で",
]

# 数量表現
quantities = [
    "多くの", "数多くの", "様々な", "多種多様な", "数百万の",
    "数十億の", "莫大な", "膨大な", "豊富な", "限られた",
]

# 形容詞
adjectives = [
    "重要な", "画期的な", "革新的な", "先進的な", "効率的な",
    "実用的な", "理論的な", "基礎的な", "応用的な", "学際的な",
    "複雑な", "単純な", "高度な", "基本的な", "根本的な",
    "新しい", "古い", "現代的な", "伝統的な", "最先端の",
]

# 比較表現
comparisons = [
    "従来の手法と比較して", "人間の能力と比べて", "他の技術と比較して",
    "以前のバージョンと比べて", "競合製品と比較して",
    "理論的な限界に近づき", "過去の記録を更新し",
]

# 技術詳細
tech_details = [
    "多層のニューラルネットワークを使用して",
    "バックプロパゲーションアルゴリズムにより",
    "勾配降下法を用いて", "確率的勾配降下法により",
    "アテンション機構を活用して", "エンコーダ・デコーダ構造で",
    "畳み込み演算を適用して", "再帰的な構造を利用して",
    "自己教師あり学習によって", "対照学習を用いて",
    "知識蒸留技術により", "プルーニング手法を適用して",
    "量子化技術を使用して", "混合精度演算により",
]


def generate_sentence():
    """ランダムな文を生成"""
    patterns = [
        # パターン1: 主語 + は + 述語
        lambda: f"{random.choice(subjects)}は{random.choice(modifiers)}{random.choice(predicates)}。",
        
        # パターン2: 時間 + 主語 + が + 述語
        lambda: f"{random.choice(time_expressions)}、{random.choice(subjects)}が{random.choice(predicates)}。",
        
        # パターン3: 理由 + 主語 + が + 述語
        lambda: f"{random.choice(reasons)}、{random.choice(subjects)}が{random.choice(predicates)}。",
        
        # パターン4: 主語 + は + 目的語述語
        lambda: (
            f"{random.choice(subjects)}は"
            f"{(op := random.choice(object_predicates))[0]}"
            f"{op[1]}。"
        ),
        
        # パターン5: 分野 + 主語 + が + 述語
        lambda: f"{random.choice(domains)}{random.choice(subjects)}が{random.choice(predicates)}。",
        
        # パターン6: 接続詞 + 主語 + は + 形容詞 + 存在だ
        lambda: f"{random.choice(conjunctions)}、{random.choice(subjects)}は{random.choice(adjectives)}存在である。",
        
        # パターン7: 数量 + 主語 + が + 分野 + 述語
        lambda: f"{random.choice(quantities)}{random.choice(subjects)}が{random.choice(domains)}{random.choice(predicates)}。",
        
        # パターン8: 比較 + 主語 + は + 述語
        lambda: f"{random.choice(comparisons)}、{random.choice(subjects)}は{random.choice(modifiers)}{random.choice(predicates)}。",
        
        # パターン9: 技術詳細 + 主語 + は + 述語
        lambda: f"{random.choice(tech_details)}、{random.choice(subjects)}は{random.choice(predicates)}。",
        
        # パターン10: 主語1 + と + 主語2 + は + 述語
        lambda: f"{random.choice(subjects)}と{random.choice(subjects)}は密接に関連している。",
        
        # パターン11: 主語 + の + 進歩により + 主語2 + が + 述語
        lambda: f"{random.choice(subjects)}の進歩により、{random.choice(subjects)}が{random.choice(predicates)}。",
        
        # パターン12: 時間 + 分野 + 主語 + 述語
        lambda: f"{random.choice(time_expressions)}、{random.choice(domains)}{random.choice(subjects)}が{random.choice(modifiers)}{random.choice(predicates)}。",
    ]
    
    return random.choice(patterns)()


def main():
    print("多様なコーパスを生成中...")
    
    # 10000行のユニークな文を生成
    sentences = set()
    target_count = 10000
    
    while len(sentences) < target_count:
        sentence = generate_sentence()
        sentences.add(sentence)
        
        if len(sentences) % 1000 == 0:
            print(f"  生成済み: {len(sentences)}/{target_count}")
    
    # ファイルに保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    
    print(f"\n✅ 完了！{len(sentences)}行を {OUTPUT_PATH.name} に保存")
    
    # サンプル表示
    print("\n📝 サンプル文:")
    sample = list(sentences)[:10]
    for s in sample:
        print(f"  {s}")


if __name__ == "__main__":
    main()
