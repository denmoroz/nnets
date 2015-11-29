# TODO: any better solution for storing intermediate data in models?

class DynamicStorage(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def for_keys(self, keys, *args):
        result = []

        for key in keys:
            node = self[key]

            for node_name in args:
                node = node[node_name]

            result.append(node)

        return result
