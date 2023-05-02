def ensure_list_string(x: "list"):
    if type(x) != list:
        raise TypeError("Not a list")
    for i in range(len(x)):
        if type(x[i]) != str:
            x[i] = str(x[i])
    return x