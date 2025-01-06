from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochPrefix:
    prefix: str
    meanings: frozenset[str]  # helper variable, we group together some of these strings

    def __repr__(self) -> str:
        return self.prefix

    @staticmethod
    def possibilities(size: int) -> list["EpochPrefix"]:
        elements: list["EpochPrefix"] = [
            EpochPrefix(prefix="", meanings=frozenset(["h" * size]))
        ]
        for n in range(size):
            meanings = []
            prefix = "h" * n + "a"
            for epoch_config in range(2 ** (size - 1 - n)):
                arr = []
                for _ in range(size - 1 - n):
                    arr.append("a" if epoch_config % 2 == 1 else "h")
                    epoch_config //= 2
                meanings.append(prefix + "".join(arr))
            element = EpochPrefix(
                prefix=prefix,
                meanings=frozenset(meanings),
            )

            elements.append(element)

        return elements


@dataclass(frozen=True)
class EpochPostfix:
    """
    Can be AA, AHA, AA, anything at the end of the string
    """

    postfix: str
    meanings: frozenset[str]  # Here we have only one hidden meaning

    def __repr__(self) -> str:
        return self.postfix

    @staticmethod
    def possibilities(size: int) -> list["EpochPostfix"]:
        elements: list["EpochPostfix"] = []

        for epoch_config in range(2**size):
            arr = []
            while epoch_config > 0:
                arr.append("a" if epoch_config % 2 == 1 else "h")
                epoch_config //= 2
            postfix = "".join(arr[::-1])
            elements.append(
                EpochPostfix(
                    postfix=postfix,
                    meanings=frozenset([(size - len(postfix)) * "h" + postfix]),
                )
            )

        return elements


@dataclass(frozen=True)
class EpochString:
    prefix: EpochPrefix
    postfix: EpochPostfix

    def __repr__(self) -> str:
        return f"{self.prefix}#{self.postfix}"

    @staticmethod
    def possibilities(size_pre: int, size_post: int) -> list["EpochString"]:
        elements = []
        for prefix in EpochPrefix.possibilities(size=size_pre):
            for postfix in EpochPostfix.possibilities(size=size_post):
                elements.append(EpochString(prefix=prefix, postfix=postfix))

        return elements


@dataclass(frozen=True)
class AttackString:
    postfix_prev: EpochPostfix
    prefix_next: EpochPrefix

    def __repr__(self) -> str:
        return f"{self.postfix_prev}.{self.prefix_next}"

    @staticmethod
    def possibilities(
        size_postf_prev: int, size_pref_next: int
    ) -> list["AttackString"]:
        elements = []
        for prefix in EpochPrefix.possibilities(size=size_pref_next):
            for postfix in EpochPostfix.possibilities(size=size_postf_prev):
                elements.append(AttackString(postfix_prev=postfix, prefix_next=prefix))

        return elements


@dataclass(frozen=True)
class ExtendedAttackString:
    attack_string: AttackString
    postfix_epoch_next: EpochPostfix

    def __repr__(self) -> str:
        return f"{self.attack_string}#{self.postfix_epoch_next}"

    def next(self, epoch_string: EpochString) -> "ExtendedAttackString":
        return ExtendedAttackString(
            attack_string=AttackString(
                postfix_prev=deepcopy(self.postfix_epoch_next),
                prefix_next=epoch_string.prefix,
            ),
            postfix_epoch_next=epoch_string.postfix,
        )

    def possibilities(
        size_prefix: int, size_postfix: int
    ) -> list["ExtendedAttackString"]:
        result: list["ExtendedAttackString"] = []
        for attack_string in AttackString.possibilities(
            size_postf_prev=size_postfix, size_pref_next=size_prefix
        ):
            for postfix_next in EpochPostfix.possibilities(size=size_postfix):
                result.append(
                    ExtendedAttackString(
                        attack_string=attack_string, postfix_epoch_next=postfix_next
                    )
                )

        return result


if __name__ == "__main__":
    print(len(ExtendedAttackString.possibilities(size_prefix=2, size_postfix=8)))
