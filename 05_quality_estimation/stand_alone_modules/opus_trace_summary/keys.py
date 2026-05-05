def parse_shard_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def make_key(model, direction_key, shard_id):
    shard_id = parse_shard_id(shard_id)
    if not model or not direction_key or shard_id is None:
        return None
    return (str(model), str(direction_key), shard_id)


def parse_state_key(value):
    parts = str(value).split("/")
    if len(parts) != 3:
        return None
    return make_key(parts[0], parts[1], parts[2])


def key_to_string(key):
    return "%s/%s/%s" % (key[0], key[1], key[2])


def pct(part, whole):
    if not whole:
        return 0.0
    return round(100.0 * float(part) / float(whole), 4)


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
