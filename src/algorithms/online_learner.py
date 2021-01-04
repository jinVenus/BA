for batch in dataloader.get():
    predict()
    if need_update == True:
        update(buffer)
        store()
        clear_buffer()
    else:
        append_buffer()
