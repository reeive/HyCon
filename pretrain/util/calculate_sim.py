# a pair of imgs or a pair of vector
# example
# criterion = nn.CrossEntropyLoss()
# loss = criterion(y_pre, y_train)
# from models import NT_Xent
# criterion = NT_Xent(args.batch_size, args.temperature, args.device, args.world_size)
# loss = criterion(z_i, z_j, i, j)
def sim(a, b, i, j):
    return 0


