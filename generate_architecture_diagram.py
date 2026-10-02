import graphviz

def create_system_diagram():
    dot = graphviz.Digraph('System_Architecture', format='png')
    dot.attr(rankdir='TB', size='10,12', dpi='300')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Sans-Serif')

    # サブグラフ：入力
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='🎤 入力・前処理層', style='filled', color='lightgrey', fillcolor='#FAFAFA')
        c.node('mic', '実機マイク / WAVファイル', fillcolor='#E0E0E0')
        c.node('preprocess', '前処理モジュール\n(32kHz/16kHz リサンプリング)', fillcolor='#E0E0E0')

    # サブグラフ：推論・検知
    with dot.subgraph(name='cluster_detection') as c:
        c.attr(label='🧠 音響検知・推論層 (realtime_detect.py)', style='filled', color='#0288D1', fillcolor='#E1F5FE')
        c.node('efficientat', 'EfficientAT (MobileNetV3)\n5クラス分類推論 (22.4ms)', fillcolor='#B3E5FC')
        c.node('fastsde', 'FAST-SDE\n単一マイク距離推定 (補助)', fillcolor='#B3E5FC')
        c.node('guard', '判定ガード・デバウンス制御\n(RMS閾値 / MARGIN閾値 / エッジトリガー)', fillcolor='#FFF9C4')

    # サブグラフ：DB
    with dot.subgraph(name='cluster_db') as c:
        c.attr(label='💾 データストア層 (db.py)', style='filled', color='#388E3C', fillcolor='#E8F5E9')
        c.node('db', 'SQLite Database\n(detections.db)', shape='cylinder', fillcolor='#C8E6C9')

    # サブグラフ：UI/フィードバック
    with dot.subgraph(name='cluster_ui') as c:
        c.attr(label='💻 UI & フィードバック層', style='filled', color='#7B1FA2', fillcolor='#F3E5F5')
        c.node('flask', 'Flask Webダッシュボード\n(frontend_app.py)', fillcolor='#E1BEE7')
        c.node('feedback', '周辺視覚ナッジ (Edge Lighting)\n& ILD左右判定表示', fillcolor='#E1BEE7')

    # エッジ接続
    dot.edge('mic', 'preprocess')
    dot.edge('preprocess', 'efficientat')
    dot.edge('preprocess', 'fastsde')
    dot.edge('efficientat', 'guard')
    dot.edge('guard', 'db', label='検知イベント保存')
    dot.edge('fastsde', 'db', label='距離データ')
    dot.edge('db', 'flask', label='ポーリング/API')
    dot.edge('db', 'feedback', label='発光トリガー')

    dot.render('/Users/Owner/Downloads/sotsuken/system_architecture', cleanup=True)
    print("構成図画像を生成しました: /Users/Owner/Downloads/sotsuken/system_architecture.png")

if __name__ == '__main__':
    create_system_diagram()