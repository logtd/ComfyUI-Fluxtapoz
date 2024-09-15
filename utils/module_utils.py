def isinstance_str(x: object, cls_name: str):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

