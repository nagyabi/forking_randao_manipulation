def ForkAction(plan: str, a_1: int, n: int) -> str | None:
    C = {i for i in range(a_1 - n + 1) if plan[i] == "C"}
    if len(C) == 0:
        return None
    l = max(C)
    plan_1 = plan[:l]
    plan_2 = plan[l:]
    act_1 = plan_1.replace("C", "P").replace("N", "M")
    act_2 = plan_2.replace("C", "H").replace("N", "M")
    return act_1 + act_2
