from dataclasses import dataclass, field

from ex_ante import ExAnte
from fork_action import ForkAction

M, N = 6, 2


@dataclass
class Vertex:
    attack_string: str
    par: set[str]
    s: int

    _edges: list["Edge"] = field(default_factory=list)


@dataclass
class Edge:
    Pr: str
    act: str

    _from: Vertex
    _to: Vertex


def BuildGr(
    alpha: float, attack_string: str, Pr: str
) -> tuple[Vertex | None, list[Vertex], list[Edge]]:
    (v, V, E) = SelfishMixingNodes(alpha, attack_string, Pr)
    if len(V) == 0:
        (v, V, E) = ForkNodes(alpha, attack_string, Pr)
    return (v, V, E)


def SelfishMixingNodes(
    alpha: float, attack_string: str, Pr: str
) -> tuple[Vertex | None, list[Vertex], list[Edge]]:
    v_b: Vertex | None = None
    V: list[Vertex] = []
    E: list[Edge] = []
    attack_string_1, attack_string_2 = attack_string.split(".")
    if attack_string_1 == "":
        v_b = Vertex(attack_string, {Pr}, 32)
        V = [v_b]
    #    elif attack_string_1[-1] == "H" and attack_string_2 == "":
    #        v_b = Vertex(attack_string, set(), 32 - len(attack_string_1))
    #        v_e, V_e, E_e = BuildGr(alpha, ".", Pr=Pr + "C" * len(attack_string_1))
    #        act = "P" * attack_string_1.count("A")
    #        e = Edge(Pr="C" * len(attack_string_1), act=act, _from=v_b, _to=v_e)
    #        v_b._edges.append(e)
    #        V = [v_b] + V_e
    #        E = [e] + E_e
    elif attack_string_1.count("H") == 0:
        (v_m, V_m, E_m) = BuildGr(alpha, attack_string[1:], Pr + "N")
        (v_p, V_p, E_p) = BuildGr(alpha, attack_string[1:], Pr + "C")
        v_b = Vertex(
            attack_string, par=v_m.par.union(v_p.par), s=32 - len(attack_string_1)
        )
        e_m = Edge(Pr="N", act="M", _from=v_b, _to=v_m)
        e_p = Edge(Pr="C", act="P", _from=v_b, _to=v_p)
        v_b._edges.extend([e_m, e_p])
        V = V_m + V_p + [v_b]
        E = E_m + E_p + [e_m, e_p]
    return (v_b, V, E)


def ForkNodes(alpha: float, attack_string: str, Pr: str):
    V: list[Vertex] = []
    E: list[Edge] = []
    attack_string_1, attack_string_2 = attack_string.split(".")
    attack_string_m = attack_string_1 + attack_string_2
    # Parsing attack_string_m
    a_1, h, a_2 = 0, 0, 0
    i = 0
    while i < len(attack_string_m) and attack_string_m[i] == "A":
        a_1 += 1
        i += 1
    while i < len(attack_string_m) and attack_string_m[i] == "H":
        h += 1
        i += 1
    while i < len(attack_string_m) and attack_string_m[i] == "A":
        a_2 += 1
        i += 1
    assert a_1 > 0 and h > 0 and a_2 > 0
    Eps = ExAnte(alpha, a_1, h, a_2, 32 - len(attack_string_1))
    as_r_1 = attack_string_1[a_1 + h :]
    as_n = as_r = as_r_1 + "." + attack_string_2
    K: set[str] = set()
    C: list[tuple[Vertex, str]] = []
    for i, b in Eps:
        as_f = attack_string_1[a_1 + h + i + 1 :] + "." + attack_string_2
        if as_f[0] == "H":
            continue
        for cfg in range(2**a_1):
            plan = bin(cfg)[2:].zfill(a_1).replace("0", "N").replace("1", "C")
            act = ForkAction(plan, a_1, b)
            if act is None:
                continue
            r = act.replace("P", "C").replace("H", "N").replace("M", "N")
            Pr_f = plan + "N" * (h + i) + "C"
            Pr_r = r + "C" * a_1
            (v_f, V_f, E_f) = BuildGr(alpha, as_f, Pr + Pr_f)
            (v_r, V_r, E_r) = BuildGr(alpha, as_r, Pr + Pr_r)
            K.update(v_f.par)
            v_i = Vertex(attack_string, v_f.par.union(v_r.par), 32 - len(as_r_1))
            V.extend([*V_f, *V_r, v_i])
            e_f = Edge(Pr=Pr_f, act="M" * i + "F", _from=v_i, _to=v_f)
            e_r = Edge(Pr=Pr_r, act="R", _from=v_i, _to=v_r)
            v_i._edges.extend([e_f, e_r])
            C.append((v_i, act))
    v_b = Vertex(attack_string, par=K, s=32 - len(attack_string_1))
    for v_i, act in C:
        e = Edge(Pr="", act=act, _from=v_b, _to=v_i)
        E.append(e)
        v_b._edges.append(e)
    (v_n, V_n, E_n) = BuildGr(alpha, as_n, Pr + "C" * (a_1 + h))
    e_n = Edge(Pr="C" * (a_1 + h), act="P" * a_1, _from=v_b, _to=v_n)
    v_b._edges.append(e_n)
    V.extend([*V_n, v_b])
    E.extend([*E_n, e_n])
    return (v_b, V, E)


def deep_print(vertex: Vertex, intend: str = ""):
    act_dict = {"P": "Propose", "M": "Miss", "H": "Hide", "F": "Fork", "R": "Regret"}

    def combine_as_and_slot(attack_string: str, slot: int) -> str:
        as_1, as_2 = attack_string.split(".")
        as_slot = 32 - len(as_1)
        assert as_slot <= slot, f"{as_slot=} {slot=}"
        index = slot - as_slot
        if slot >= 32:
            return as_1 + "|." + as_2
        return attack_string[:index] + "|" + attack_string[index:]

    print(f"{intend}{combine_as_and_slot(vertex.attack_string, vertex.s)}")
    for kn in vertex.par:
        print(f"{intend}known: {kn}")
    print(f"{intend}===========")
    new_intend = intend + "    "
    for edge in vertex._edges:
        act = ".".join([act_dict[a] for a in edge.act])
        print(f'{new_intend}Action={act} Printing="{edge.Pr}"')
        deep_print(edge._to, new_intend)


if __name__ == "__main__":
    attack_string = "AH.A"
    graph, V, _ = BuildGr(0.201, attack_string=attack_string, Pr="")
    for v in V:  # Compensation for Pr="", because |Pr| + |as_1|=M should hold
        v.par = {p[: len(attack_string.split(".")[0])] for p in v.par}
    deep_print(graph)
