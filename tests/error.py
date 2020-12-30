def get_error(func):
    try:
        func()
        return None
    except Exception as error:
        return error