FUNCTION max_pool(in_tensor, pool_size):
    padded_tensor = padding(in_tensor, pool_size)
    batch_size, channels, height, width = get_dimensions(padded_tensor)

# calculate ouptput tensor dimensions
    nheight = height // pool_size
    nwidth = width // pool_size
    out_tensor = create_empty_tensor(batch_size, channels, nheight, nwidth)

    FOR b FROM 0 TO batch_size - 1:
        FOR c FROM 0 TO channels - 1:
# scanning the batches and each of the channels
            FOR row FROM 0 TO height - 1 STEP pool_size:
                FOR col FROM 0 TO width - 1 STEP pool_size:
# moving across each map in pool size steps
                    window = padded_tensor[b, c, row : row + pool_size, col : col + pool_size]
# making a window of the dimentions of the pool
                    max = get_max_value(window)
                    out_tensor[b,c, row // pool_size, col // pool_size] = max
# finding the max value in the window and placing it in the output tensor
    RETURN out_tensor