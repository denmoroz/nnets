def mini_batch_iterator(data, batch_size=16):
    for k in xrange(0, len(data), batch_size):
        yield data[k:k+batch_size]