from igibson.object_states import Inside, OnTop, Under

OBJECTS_INFO = {
    ### Support objects
    "shelf": {
        # 'f000edc1cfdeda11bee0494534c13f8c', # Mostly solid
        # 'e9850d3c5243cc60f62af8489541547b', # Very wide
        "de3b28f255111570bc6a557844fbbce9": {"scale": (2, 2, 1.5)},
        # 'b7697d284c38fcade76bc197b3a3ffc0',
        "b079feff448e925546c4f23965b7dd40": {"scale": (2, 1.25, 1.5)},
        "71b87045c8fbce807c7680c7449f4f61": {"scale": None},  #
        # "6d5e521ebaba489bd9a347e62c5432e": {"scale": (2, 3, 2.5)}, # Too small ratio of cabinet space to height
        # '6ae80779dd34c194c664c3d4e2d59341', # Fails inside checking
        # '50fea70354d4d987d42b9650f19dd425', # Pointy top
        "3bff6c7c4ab1e47e2a9a1691b6f98331": {"scale": (1.5, 2, 2)},
        "38be55deba61bcfd808d50b6f71a45b": {"scale": (2, 1.5, 2)},
        "1170df5b9512c1d92f6bce2b7e6c12b7": {"scale": None},  #
    },
    "breakfast_table": {
        "1b4e6f9dd22a8c628ef9d976af675b86": {"scale": (1, 1, 1.5)},
        "5f3f97d6854426cfb41eedea248a6d25": {"scale": (1, 1, 2)},
        "33e4866b6db3f49e6fe3612af521500": {"scale": (1, 1, 2)},
        # "72c8fb162c90a716dc6d75c6559b82a2": {} #Not under sampleable
        "242b7dde571b99bd3002761e7a3ba3bd": {"scale": (1, 1, 1)},
        # "26073": {"scale": None} # Not work
        # "db665d85f1d9b1ea5c6a44a505804654": {"scale": (1, 1, 2)}, # Unstable
    },
    ### Choice objects
    "apple": {
        "00_0": {"scale": None},
        "00_2": {"scale": None},
    },
    "bowl": {
        "5aad71b5e6cb3967674684c50f1db165": {"scale": None},
        "7d7bdea515818eb844638317e9e4ff18": {"scale": None},
        "8b90aa9f4418c75452dd9cc5bac31c96": {"scale": None},
        "68_1": {"scale": None},
        "56803af65db53307467ca2ad6571afff": {"scale": None},
    },
}

# Number of extra object we import into the scene for each model.
NUM_EXTRA_OBJECTS_EACH = 5

# Total number of extra objects we place into the scene.
NUM_EXTRA_OBJECTS_TOTAL = 10

EXTRA_OBJECTS = {
    "bell_pepper": [
        "26_0",
        "26_1",
        "26_2",
        "bell_pepper_000",
    ],
}
