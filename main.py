from flask import Flask, request, jsonify
from flask_cors import CORS

import MicroTokenizer

tokenizer_loader = MicroTokenizer.load('core_pd_md')
tokenizer = tokenizer_loader.get_tokenizer()

app = Flask(__name__, static_url_path='/Users/howl/WebstormProjects/NLP_server_frontend')
app.config['JSON_AS_ASCII'] = False
CORS(app)


all_tokenizer_class = {
    'DAG': ('基于有向无环图的分词方法', tokenizer.cut_by_DAG),
    'HNM': ('基于隐马尔科夫模型的分词方法', tokenizer.cut_by_HMM),
    'CRF': ('基于条件随机场的分词方法', tokenizer.cut_by_CRF),
    'max_match_forward': ('基于最大正向匹配的分词方法', tokenizer.cut_by_max_match_forward),
    'max_match_backward': ('基于最大反向匹配的分词方法', tokenizer.cut_by_max_match_backward),
    'max_match_bidirectional': ('基于最大双向匹配的分词方法', tokenizer.cut_by_max_match_bidirectional)
}


@app.route("/single_tokenizer", methods=['GET'])
def single_tokenizer():
    tokenizer_class = request.args.get('tokenizer_class')
    message = request.args.get('message')

    if tokenizer_class not in all_tokenizer_class:
        # TODO
        raise ValueError()

    tokenizer_class_object = all_tokenizer_class[tokenizer_class][1]
    segment_result = tokenizer_class_object(message)

    return jsonify(segment_result)


@app.route("/list_tokenizer", methods=['GET'])
def list_tokenizer():
    tokenizer_info = {k: v[0] for k, v in all_tokenizer_class.items()}
    return jsonify(tokenizer_info)


if __name__ == "__main__":
    app.run()
