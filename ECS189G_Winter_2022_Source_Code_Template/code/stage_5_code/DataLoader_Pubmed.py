import pandas as pd

data_path = "/Users/jacquelinemitchell/Documents/ECS189G/sample_code/" \
            "ECS189G-21W/ECS189G_Winter_2022_Source_Code_Template/data/stage_5_data/pubmed"

# Node == paper
# Edge: source cites target
column_names_node = ["Node"] + [f"word_{i}" for i in range(500)] + ["class"]
node_data = pd.read_csv(data_path + "/node", sep="\t", engine="python", header=None, names=column_names_node)
print(node_data.head())

column_names_edges = ["target", "source"]
edge_data = pd.read_csv(data_path + "/link", sep="\t", engine="python", header=None, names=column_names_edges)
edge_data['label'] = 'cites'
print(edge_data.head())