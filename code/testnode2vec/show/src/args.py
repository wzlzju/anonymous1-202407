configs = {
    "line-0.2": {"dependence": ["default", "draw"],
        "graph_path": "line-0.2.edgelist",
        "embs_path": "line_classification_embs.json",
        "origin_path": "line.edgelist",
        "nodedata_path": "line-nodedata.pickle",
        "links_path": "line-ring-links.pickle",
        "sampling_r": 0.001,     # 0.001, 0.002
        "layout_method": "multilayer",
        "layout_arrangement": "from_position",
    },
    "ring-0.2": {"dependence": ["default", "draw"],
        "graph_path": "ring-0.2.edgelist",
        "embs_path": "ring_classification_embs.json",
        "origin_path": "ring.edgelist",
        "nodedata_path": "ring-nodedata.pickle",
        "sampling_r": 0.0015,     # 0.001, 0.002
        "layout_method": "multilayer",
        "layout_arrangement": "from_position",
    },

    "highlight_maxdegree_2_nodes": {"dependence": None,
        "highlight_nodes_method": "max_degree",
        "highlight_nodes_paras": [0,1],
    },
    "700-SUGRL": {"dependence": ["default", "draw", "highlight_maxdegree_2_nodes"],
        "data_path": "",
        "graph_path": "./testnode2vec/struc2vec-master/graph/700.edgelist",
        "embs_path": "./testnode2vec/SUGRL-test/embs/700_classification_embs.json",
        "sampling_r": 0.000088,     # 0.000088, 0.0001, 0.00015
        "layout_arrangement": "from_position",
    },
    "700-node2vec": {"dependence": ["default", "draw", "highlight_maxdegree_2_nodes"],
        "data_path": "",
        "graph_path": "./testnode2vec/struc2vec-master/graph/700.edgelist",
        "embs_path": "./testnode2vec/node2vec-master/emb/700.emb",
        "sampling_r": 0.002,
        "layout_arrangement": "from_position",
    },
    "700-struc2vec": {"dependence": ["default", "draw", "highlight_maxdegree_2_nodes"],
        "data_path": "",
        "graph_path": "./testnode2vec/struc2vec-master/graph/700.edgelist",
        "embs_path": "./testnode2vec/struc2vec-master/emb/700-2d.emb",
        "sampling_r": 0.0007,
        "layout_arrangement": "from_position",
    },


    "default": {"dependence": None,
        "data_path": "./testnode2vec/show/data/",
        "random_state": 88,      # 88, 8888888
        "clu_method": "dbscan",
        "clu_args": {"eps": 1.2},       # {"n_components": 5}
        "highlight_nodes_method": None,
        "highlight_nodes_paras": None,
    },
    "draw": {"dependence": None,
        "new_layout": False,
        "layout_method": "default",
        "layout_arrangement": "default",
        "alpha": 0.75,
        "node_size": 40, 
        "edge_color": (0.7,0.7,0.7,0.1),
    },
}

pargs = {
    "sp": {},

    ('a1', '1', 's'): 0.92,
    ('a1', '2', 's'): 1.0,
    ('a1', '3', 's'): 1.04,
    ('a1', '4', 's'): 1.0,
    ('a1', '1', 'r'): 0.9,
    ('a1', '3', 'r'): 1.1,

    ('a2', '1', 's'): 0.975,
    ('a2', '2', 's'): 1.19,
    ('a2', '3', 's'): 1.02,
    ('a2', '4', 's'): 1.02,

    ('a3', '1', 's'): 0.97,
    ('a3', '2', 's'): 1.0,
    ('a3', '3', 's'): 1.0,
    ('a3', '4', 's'): 0.76,

    ('a4', '1', 's'): 0.95,
    ('a4', '2', 's'): 1.1,
    ('a4', '3', 's'): 1.0,
    ('a4', '4', 's'): 0.9,
}

def parse_args(configs, config):
    args = {}
    if configs[config]["dependence"]:
        for parent_config in configs[config]["dependence"]:
            args.update(parse_args(configs, parent_config))
    args.update(configs[config])
    return args


if __name__ == "__main__":
    print(parse_args(configs, "default"))
    print(parse_args(configs, "line-0.2"))
    print(parse_args(configs, "700-SUGRL"))