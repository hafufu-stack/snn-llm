"""
Japanese Corpus Downloader
==========================

Downloads and processes Japanese text from:
1. Aozora Bunko (青空文庫) - Public domain literature
2. Wikipedia excerpts (if available)

Usage: python experiments/download_corpus.py

Author: ろーる
Date: 2026-01-22
"""

import os
import re
import urllib.request
import zipfile
import io
from typing import List
import time

# 青空文庫の人気作品（著作権切れ）
AOZORA_WORKS = [
    # (作品ID, ZIPファイルURL, タイトル)
    ("1567", "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip", "こころ（夏目漱石）"),
    ("752", "https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip", "坊っちゃん（夏目漱石）"),
    ("758", "https://www.aozora.gr.jp/cards/000148/files/758_ruby_5638.zip", "草枕（夏目漱石）"),
    ("772", "https://www.aozora.gr.jp/cards/000081/files/456_ruby_145.zip", "銀河鉄道の夜（宮沢賢治）"),
    ("1779", "https://www.aozora.gr.jp/cards/000035/files/301_ruby_41993.zip", "人間失格（太宰治）"),
    ("1563", "https://www.aozora.gr.jp/cards/000035/files/2287_ruby_2670.zip", "走れメロス（太宰治）"),
    ("8", "https://www.aozora.gr.jp/cards/000879/files/127_ruby_936.zip", "蜘蛛の糸（芥川龍之介）"),
    ("24", "https://www.aozora.gr.jp/cards/000879/files/33_ruby_1073.zip", "羅生門（芥川龍之介）"),
    ("625", "https://www.aozora.gr.jp/cards/000879/files/179_ruby_5765.zip", "鼻（芥川龍之介）"),
    ("234", "https://www.aozora.gr.jp/cards/000042/files/234_ruby_20.zip", "風立ちぬ（堀辰雄）"),
]


def remove_ruby(text: str) -> str:
    """ルビ（振り仮名）を除去"""
    # 《xxx》形式のルビを除去
    text = re.sub(r'《[^》]*》', '', text)
    # ［＃xxx］形式の注記を除去
    text = re.sub(r'［＃[^］]*］', '', text)
    # ｜記号を除去
    text = text.replace('｜', '')
    return text


def clean_text(text: str) -> str:
    """テキストのクリーニング"""
    # ルビ除去
    text = remove_ruby(text)
    
    # 連続する空白・改行を整理
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ 　]{2,}', ' ', text)
    
    # 前後の空白を削除
    text = text.strip()
    
    return text


def download_aozora(work_info: tuple) -> str:
    """青空文庫からテキストをダウンロード"""
    work_id, url, title = work_info
    
    try:
        print(f"  Downloading: {title}")
        
        # ZIPファイルをダウンロード
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (SNN-LLM Research)'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            zip_data = response.read()
        
        # ZIPを解凍してテキストを抽出
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            for filename in zf.namelist():
                if filename.endswith('.txt'):
                    # Shift-JISでエンコードされている
                    text = zf.read(filename).decode('shift_jis', errors='ignore')
                    return clean_text(text)
        
        return ""
    
    except Exception as e:
        print(f"    Error downloading {title}: {e}")
        return ""


def create_ai_corpus() -> str:
    """AI・脳科学関連のコーパス（自作）"""
    return """
脳科学とAIは密接な関係にあります。人間の脳は約860億個のニューロンで構成されており、
これらが複雑なネットワークを形成して思考や記憶を可能にしています。

スパイキングニューラルネットワーク（SNN）は、この生物学的なニューロンの動作を
模倣したAIモデルです。従来のディープラーニングとは異なり、SNNは時間的な情報を
活用することで、より効率的な計算が可能です。

Blue Brain Projectの研究によると、人間の脳には11次元のトポロジー構造が
存在することがわかっています。この発見は、効率的な情報処理の鍵となる可能性があります。

言語モデルは、次の単語を予測することで文章を生成します。
GPTやBERTなどの大規模言語モデルは数十億のパラメータを持ち、
膨大な計算資源を必要とします。

しかし、SNNを活用することで、同等の性能を遥かに少ないエネルギーで
達成できる可能性があります。これは、エッジデバイスでのAI処理や、
持続可能なAI開発において重要な意味を持ちます。

脳型コンピューティングの研究は、より効率的で人間に近いAIの実現を目指しています。
ニューロモーフィックチップやスパイキングニューラルネットワークは、
この目標に向けた重要な技術です。

日本語は、ひらがな、カタカナ、漢字の3種類の文字体系を持つ複雑な言語です。
文法構造も英語とは大きく異なり、主語-目的語-動詞の語順を持ちます。
このような特性は、言語モデルの設計において考慮すべき重要な要素です。

自然言語処理の分野では、形態素解析やトークナイゼーションが基本的な処理として重要です。
日本語では、MeCabやJumanなどの形態素解析器が広く使用されています。

機械学習モデルの学習には、大量のトレーニングデータが必要です。
データの質と量は、モデルの性能に直接影響を与えます。

ハイパーパラメータの調整は、モデル開発において重要なプロセスです。
学習率、バッチサイズ、エポック数などのパラメータを適切に設定することで、
モデルの性能を最大化できます。

転移学習は、事前学習されたモデルを新しいタスクに適応させる技術です。
これにより、少量のデータでも高い性能を達成することができます。

注意機構（Attention）は、Transformerアーキテクチャの核心となる技術です。
入力シーケンスの重要な部分に「注意」を向けることで、
長距離依存関係を効果的に捉えることができます。

再帰的ニューラルネットワーク（RNN）やLSTMは、
系列データの処理に広く使用されてきました。
しかし、Transformerの登場により、多くのタスクでこれらを上回る性能が達成されています。

量子コンピューティングは、次世代の計算パラダイムとして注目されています。
量子ビットの重ね合わせと量子もつれを活用することで、
特定の問題に対して指数関数的な高速化が可能です。

エッジAIは、クラウドに依存しないローカルな推論処理を可能にします。
プライバシーの保護やリアルタイム処理の観点から、重要性が増しています。

持続可能なAI開発は、環境への影響を考慮した技術開発を目指しています。
エネルギー効率の高いモデルアーキテクチャの研究は、この観点から重要です。
"""


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         📚 Japanese Corpus Downloader for SNN-LLM             ║
    ║         青空文庫 + AI/脳科学コーパス                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    all_texts = []
    
    # 1. 青空文庫からダウンロード
    print("\n[1/3] Downloading from Aozora Bunko (青空文庫)...")
    for work_info in AOZORA_WORKS:
        text = download_aozora(work_info)
        if text:
            all_texts.append(text)
            print(f"    ✓ {len(text):,} chars")
        time.sleep(1)  # サーバー負荷軽減
    
    # 2. AI・脳科学コーパス
    print("\n[2/3] Adding AI/Brain Science corpus...")
    ai_corpus = create_ai_corpus()
    all_texts.append(ai_corpus * 10)  # 10回繰り返し
    print(f"    ✓ {len(ai_corpus) * 10:,} chars")
    
    # 3. 既存のコーパスがあれば追加
    existing_path = os.path.join(data_dir, "japanese_corpus.txt")
    if os.path.exists(existing_path):
        print("\n[3/3] Merging with existing corpus...")
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing = f.read()
            all_texts.append(existing)
            print(f"    ✓ {len(existing):,} chars")
    
    # 統合
    full_corpus = "\n\n".join(all_texts)
    
    # 保存
    output_path = os.path.join(data_dir, "japanese_corpus_large.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_corpus)
    
    # 統計
    n_chars = len(full_corpus)
    n_lines = full_corpus.count('\n')
    size_mb = n_chars / 1024 / 1024
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    ダウンロード完了！🎉                       ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   文字数:     {n_chars:>12,} chars                            ║
    ║   行数:       {n_lines:>12,} lines                            ║
    ║   サイズ:     {size_mb:>12.2f} MB                             ║
    ║   出力:       {output_path:<40}║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
