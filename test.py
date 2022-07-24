test = [[ 344, 761],
        [ 388, 106],
        [1064, 152],
        [1020, 807]]

test = sorted(test, key=lambda x:x[0])
l, r = test[:2], test[2:]
l = sorted(l, key=lambda x:x[1])
r = sorted(r, key=lambda x:x[1])
print([l[0], r[0], r[1], l[1]])

