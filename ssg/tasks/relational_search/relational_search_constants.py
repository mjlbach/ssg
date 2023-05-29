from igibson.object_states import (Burnt, Cooked, Dusty, Inside, OnTop,
                                   Soaked, Stained, Under)
from ssg.utils.other_utils import convert_to_onehot
from ssg.utils.room_constants import ROOMS

SEARCH_OBJECTS = {
    "apple": ["00_0", "00_2"],
    "bath_towel": [
        "Provence_Bath_Towel_Royal_Blue",
        "Threshold_Performance_Bath_Sheet_Sandoval_Blue",
        "Threshold_Textured_Damask_Bath_Towel_Pink",
    ],
    "bell_pepper": [
        "26_0",
        "26_1",
        "26_2",
        "bell_pepper_000",
    ],
    "bowl": [
        "260545503087dc5186810055962d0a91",
        "56803af65db53307467ca2ad6571afff",
        "5aad71b5e6cb3967674684c50f1db165",
        "6494761a8a0461777cba8364368aa1d",
        "68_0",
        "68_1",
        "68_2",
        "68_3",
        "7d7bdea515818eb844638317e9e4ff18",
        "80_0",
        "8b90aa9f4418c75452dd9cc5bac31c96",
        "a1393437aac09108d627bfab5d10d45d",
        "c25fd49b75c12ef86bbb74f0f607cdd",
    ],
    "gym_shoe": [
        "adistar_boost_yellow_grey",
        "F5_TRX_FG_adidas_soccer_show_yellow",
        "FYW_DIVISION_adidas_basketball_shoe_grey",
        "TERREX_FAST_blue_sneaker",
        "TOP_TEN_HI_red_sneaker_shoe",
        "TZX_Runner_adidas_purple_grey_shoe",
        "ZigKick_Hoops_reebok_blue_white_show",
        "ZX700_adidas_black_orange_sneaker",
        "ZX700_adidas_purple_yellow_show",
        "ZX700_adidas_white_red_sneaker",
    ],
    "hat": [
        "DPC_Handmade_Hat_Brown",
        "DPC_tropical_Trends_straw_Hat",
        "Retail_Leadership_Summit_Grey_Trilby",
        "Retail_Leadership_Summit_Straw_Hat",
    ],
    "paper_towel": [
        "33_0",
    ],
    "rag": [
        "rag_000",
        "rag_001",
    ],
    "tomato": [
        "16_0",
        "16_1",
    ],
}

SEARCH_CATEGORY_STATES = {
    "apple": [Burnt, Cooked],
    "bath_towel":  [Soaked],
    "bell_pepper":  [Burnt, Cooked],
    "bowl":  [Dusty, Stained],
    "gym_shoe": [Dusty, Stained],
    "hat": [Dusty, Stained],
    "paper_towel": [Soaked],
    "rag": [Soaked],
    "tomato": [Burnt, Cooked],
}

OBJ_STATES = {
    "Normal": None,
    "Burnt": Burnt,
    "Cooked": Cooked,
    "Soaked": Soaked,
    "Dusty": Dusty,
    "Stained": Stained,
}

RELATIONAL_STATES = {
    "OnTop": OnTop,
    "Under": Under,
    "Inside": Inside,
}

# Ignoring these for now, but possible?

OBJ_SAMPLING_STATES = {
    "breakfast_table": [OnTop, Under],
    "bottom_cabinet": [OnTop],
    "countertop": [OnTop],
    "console_table": [OnTop, Under],
    "coffee_table": [OnTop, Under],
    "shelf": [OnTop, Inside],
    "stove": [OnTop],
    "washer": [OnTop],
    "dryer": [OnTop],
    "pool_table": [OnTop, Under],
    "swivel_chair": [OnTop],
    # "straight_chair": [OnTop, Under],
    "armchair": [OnTop],
    "stool": [OnTop, Under],
    "trash_can": [Inside],
    "treadmill": [OnTop],
    # "sofa": [OnTop],
    "bathtub": [Inside],
    "chest": [OnTop],
    "crib": [Inside],
    "shower": [Inside],
    "piano": [OnTop],
}

UNIFIED_CATEGORICAL_ENCODING = convert_to_onehot(list(SEARCH_CATEGORY_STATES) + list(OBJ_SAMPLING_STATES) + list(ROOMS) + ["other"])
