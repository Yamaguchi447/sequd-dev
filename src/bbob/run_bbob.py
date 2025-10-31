import bbob.bbobbenchmarks as bn


def select_bbo(func_id, seed, X_opt, F_opt):

    if func_id == 1:
        f = bn.F1(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 2:
        f = bn.F2(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 3:
        f = bn.F3(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 4:
        f = bn.F4(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 5:
        f = bn.F5(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 6:
        f = bn.F6(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 7:
        f = bn.F7(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 8:
        f = bn.F8(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 9:
        f = bn.F9(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 10:
        f = bn.F10(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 11:
        f = bn.F11(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 12:
        f = bn.F12(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 13:
        f = bn.F13(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 14:
        f = bn.F14(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 15:
        f = bn.F15(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 16:
        f = bn.F16(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 17:
        f = bn.F17(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 18:
        f = bn.F18(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 19:
        f = bn.F19(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 20:
        f = bn.F20(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 21:
        f = bn.F21(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 22:
        f = bn.F22(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 23:
        f = bn.F23(seed, zerox=not X_opt, zerof=not F_opt)
    elif func_id == 24:
        f = bn.F24(seed, zerox=not X_opt, zerof=not F_opt)
    return f


def eval(x, func_id, seed, X_opt=True, F_opt=True):
    f = select_bbo(func_id, seed, X_opt, F_opt)
    val = f(x)
    return val
