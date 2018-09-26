import copy

from flask import Flask, request, jsonify
from flask_cors import CORS

import MicroTokenizer

tokenizer_loader = MicroTokenizer.load('core_pd_md')
tokenizer = tokenizer_loader.get_tokenizer()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['DEBUG'] = True
CORS(app)


all_tokenizer_class = {
    'DAG': ('基于有向无环图的分词方法', tokenizer.cut_by_DAG, tokenizer.dag_tokenizer.graph_builder),
    'HNM': ('基于隐马尔科夫模型的分词方法', tokenizer.cut_by_HMM, None),
    'CRF': ('基于条件随机场的分词方法', tokenizer.cut_by_CRF, None),
    'max_match_forward': ('基于最大正向匹配的分词方法', tokenizer.cut_by_max_match_forward, tokenizer.max_match_forward_tokenizer),
    'max_match_backward': ('基于最大反向匹配的分词方法', tokenizer.cut_by_max_match_backward, tokenizer.max_match_backward_tokenizer),
    'max_match_bidirectional': ('基于最大双向匹配的分词方法', tokenizer.cut_by_max_match_bidirectional, None)
}

dict_based_tokenizer = {
    k: v for k, v in all_tokenizer_class.items() if k in ('DAG', 'max_match_forward', 'max_match_backward')
}


@app.route("/single_tokenizer", methods=['GET'])
def single_tokenizer():
    tokenizer_class = request.args.get('tokenizer_class')
    message = request.args.get('message')

    if tokenizer_class not in all_tokenizer_class:
        # TODO
        raise ValueError()

    tokenizer_func = all_tokenizer_class[tokenizer_class][1]
    segment_result = tokenizer_func(message)

    return jsonify(segment_result)


def parse_custom_dict(custom_dict_str):
    if not custom_dict_str:
        return []

    custom_dict = []

    for i in custom_dict_str.split('\n'):
        token_plus = i.split()
        if len(token_plus) > 1:
            token, weight = token_plus
            token = str(token)
            weight = int(weight)
        else:
            token = str(token_plus[0])
            weight = 1

        custom_dict.append((token, weight))

    print(custom_dict)

    return custom_dict


@app.route("/single_tokenizer_with_custom_dict", methods=['GET'])
def single_tokenizer_with_custom_dict():
    tokenizer_class = request.args.get('tokenizer_class')
    message = request.args.get('message')
    custom_dict_str = request.args.get('custom_dict')

    custom_dict = parse_custom_dict(custom_dict_str)

    if tokenizer_class not in dict_based_tokenizer:
        # TODO
        raise ValueError()

    tokenizer_func = all_tokenizer_class[tokenizer_class][1]
    tokenizer_class_object = all_tokenizer_class[tokenizer_class][2]

    dict_data = copy.deepcopy(tokenizer_class_object.dict_data)
    origin_dict_data = tokenizer_class_object.dict_data

    for token, weight in custom_dict:
        dict_data.add_token_and_weight(token, weight)

    # assign dict_data to tokenizer
    tokenizer_class_object.dict_data = dict_data

    segment_result = tokenizer_func(message)

    # restore dict_data to origin
    tokenizer_class_object.dict_data = origin_dict_data

    return jsonify(segment_result)


@app.route("/tokenizer_with_fusion", methods=['GET'])
def tokenizer_with_fusion():
    tokenizer_class_list = request.args.getlist('tokenizer_class_list[]')
    print(tokenizer_class_list)
    message = request.args.get('message')

    if all(i not in all_tokenizer_class for i in tokenizer_class_list):
        # TODO
        raise ValueError()

    solutions = []
    for tokenizer_class in tokenizer_class_list:
        tokenizer_func = all_tokenizer_class[tokenizer_class][1]
        segment_result = tokenizer_func(message)
        solutions.append(segment_result)

    best_solution = tokenizer.joint_solutions(solutions)

    return jsonify(best_solution)


@app.route("/list_tokenizer", methods=['GET'])
def list_tokenizer():
    tokenizer_info = {k: v[0] for k, v in all_tokenizer_class.items()}
    return jsonify(tokenizer_info)


@app.route("/list_dict_based_tokenizer", methods=['GET'])
def list_dict_based_tokenizer():
    tokenizer_info = {k: v[0] for k, v in dict_based_tokenizer.items()}
    return jsonify(tokenizer_info)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
