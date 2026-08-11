from panphon.distance import Distance

distance = Distance(feature_model="strict")


def phonetic_distance(a, b):
    return distance.weighted_feature_edit_distance(a, b)

print(phonetic_distance("tova", "gava"))