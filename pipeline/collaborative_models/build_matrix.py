from pipeline.config_base import ConfigBase
import pandas as pd
from typing import Dict, Tuple
from scipy.sparse import coo_matrix


def get_user_item_matrix(config: ConfigBase, item_column="story_id") -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    reaction = pd.read_csv(config.stories_path)

    view2marks: Dict[str, float] = {
        "dislike": 1.,
        "skip": 2.,
        "view": 4.,
        "like": 5.
    }

    reaction["result"] = [view2marks[event] for event in reaction["event"]]

    user_mapper: Dict[int, int] = dict()

    for i, customer in enumerate(reaction["customer_id"].unique().tolist()):
        user_mapper[customer] = i

    item_mapper: Dict[int, int] = dict()

    for i, item in enumerate(reaction[item_column].unique().tolist()):
        item_mapper[item] = i

    reaction["u_index"] = [user_mapper[user] for user in reaction["customer_id"]]
    reaction["i_index"] = [item_mapper[item] for item in reaction[item_column]]

    user_item_matrix = coo_matrix(
        (reaction["result"],
        (reaction["u_index"], reaction["i_index"])),
    )

    return user_item_matrix[["u_index", "i_index", "result"]], user_mapper, item_mapper