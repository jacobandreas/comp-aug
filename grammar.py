from collections import namedtuple
import numpy as np

NTRule = namedtuple('NTRule', ['lhs', 'rhs', 'i_e', 'i_f'])
TRule = namedtuple('TRule', ['lhs', 'rhs_e', 'rhs_f'])
class Grammar(object):
    def __init__(self, root, rules):
        self.root = root
        self.rules = rules

    def sample(self, lhs=None):
        if lhs is None:
            lhs = self.root
        rules = self.rules[lhs]
        rule = rules[np.random.randint(len(rules))]
        if isinstance(rule, TRule):
            return (rule.rhs_e,), (rule.rhs_f,)
        elif isinstance(rule, NTRule):
            produced = [self.sample(s) for s in rule.rhs]
            e = sum([produced[i][0] for i in rule.i_e], ())
            f = sum([produced[i][1] for i in rule.i_f], ())
            return e, f

    def prune(self):
        visited = {nt: False for nt in self.rules.keys()}
        generates = {
            nt: any(isinstance(r, TRule) for r in rr) 
            for nt, rr in self.rules.items()
        }
        keep_rules = {}
        def check(nt):
            if visited[nt]:
                return generates[nt]
            visited[nt] = True
            usable = [
                r for r in self.rules[nt]
                if isinstance(r, TRule) or all(check(n) for n in r.rhs)
            ]
            generates[nt] = generates[nt] or len(usable) > 0
            keep_rules[nt] = usable
            return generates[nt]

        check(self.root)
        keep_rules = {
            nt: rr 
            for nt, rr in keep_rules.items()
            if visited[nt] and generates[nt]
        }
        return Grammar(self.root, keep_rules)

class GrammarBuilder(object):
    def symbols(self, n_t):
        return [chr(ord('a') + i) for i in range(n_t)]

    def _try_sample(self, n_nt, n_t, n_nt_rules, n_t_rules):
        assert n_nt > 1
        assert n_t > 1
        nt_names = [chr(ord('A') + i) for i in range(n_nt)]
        t_e_names = [chr(ord('a') + i) for i in range(n_t)]
        t_f_names = [chr(ord('a') + i) for i in range(n_t)]
        #t_e_names = list(range(n_t))
        #t_f_names = list(range(n_t))

        i_e = (0, 1)
        i_f = (0, 1) if np.random.randint(2) == 0 else (1, 0)

        nt_rules = []
        for _ in range(n_nt_rules):
            lhs = np.random.randint(n_nt - 1)
            rhs1 = lhs + 1 + np.random.randint(n_nt - lhs - 1)
            rhs2 = lhs + 1 + np.random.randint(n_nt - lhs - 1)
            nt_rules.append(NTRule(
                nt_names[lhs],
                (nt_names[rhs1], nt_names[rhs2]),
                i_e,
                i_f
            ))

        t_rules = []
        for _ in range(n_t_rules):
            lhs = np.random.randint(n_nt)
            rhs_e = np.random.randint(n_t)
            rhs_f = np.random.randint(n_t)
            t_rules.append(TRule(
                nt_names[lhs],
                t_e_names[rhs_e],
                t_f_names[rhs_f]
            ))

        rules = {nt: [] for nt in nt_names}
        for rule in nt_rules:
            rules[rule.lhs].append(rule)
        for rule in t_rules:
            rules[rule.lhs].append(rule)

        grammar = Grammar(nt_names[0], rules)
        grammar = grammar.prune()
        if len(grammar.rules) == 0 or grammar.root not in grammar.rules:
            raise RuntimeError('sampled a bad grammar')
        return grammar

    def sample(self, n_nt, n_t, n_nt_rules, n_t_rules):
        counter = 0
        while counter < 100:
            try:
                return self._try_sample(n_nt, n_t, n_nt_rules, n_t_rules)
            except RuntimeError as e:
                counter += 1
        raise RuntimeError(
            "Couldn't generate a good grammar in 100 attempts - try adjusting "
            "sampler parameters"
        )
