from qsim.graph_algorithms.graph import line_graph, ring_graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qsim.tools.tools import equal_superposition
from qsim.evolution import hamiltonian
from qsim.graph_algorithms import qaoa
import re


def multiply_pauli(p1, p2):
    if p1 == 'x':
        if p2 == 'x':
            return 1, 'i'
        elif p2 == 'y':
            return 1j, 'z'
        elif p2 == 'z':
            return -1j, 'y'
        elif p2 == 'i':
            return 1, 'x'
        else:
            raise Exception

    elif p1 == 'y':
        if p2 == 'x':
            return -1j, 'z'
        elif p2 == 'y':
            return 1, 'i'
        elif p2 == 'z':
            return 1j, 'x'
        elif p2 == 'i':
            return 1, 'y'
        else:
            raise Exception

    elif p1 == 'z':
        if p2 == 'x':
            return 1j, 'y'
        elif p2 == 'y':
            return -1j, 'x'
        elif p2 == 'z':
            return 1, 'i'
        elif p2 == 'i':
            return 1, 'z'
        else:
            raise Exception
    elif p1 == 'i':
        return 1, p2
    else:
        raise Exception


def fmt_coeff(coeff):
    if coeff == 1:
        return ''
    elif coeff == 1j:
        return '[1j]'
    elif coeff == -1j:
        return '[-1j]'
    elif coeff == -1:
        return '[-1]'
    else:
        return str(coeff)


def heisenberg_operator(edge, graph, depth):
    initial_string = 'i' * edge[0] + 'z' + 'i' * (edge[1] - edge[0] - 1) + 'z' + (
            graph.number_of_nodes() - edge[1] - 1) * 'i'

    terms = {initial_string: ''}

    def increment_power(matchobj):
        return matchobj.group()[:3] + str(int(matchobj.group()[3]) + 1) + matchobj.group()[4:]

    def commute_with_cost(term):
        z_and_i = (term[edge[0]] == 'z' or term[edge[0]] == 'i') and (term[edge[1]] == 'z' or term[edge[1]] == 'i')
        x_and_y = (term[edge[0]] == 'x' or term[edge[0]] == 'y') and (term[edge[1]] == 'x' or term[edge[1]] == 'y')
        return z_and_i or x_and_y

    subscript = 0

    for p in range(1, depth+1):
        if p % 2 == 1:  # If current depth is odd
            subscript += 1
            # Apply mixer Hamiltonian
            for node in graph.nodes:
                new_terms = {key: '' for key in terms}
                #print(new_terms)
                #print(terms)
                for term in terms:
                    if term[node] != 'x' and term[node] != 'i':  # Operator does not commute with x
                        new_term_cos = ''
                        for subterm in terms[term].split('+'):
                            # Implement cos
                            new_power, n = re.subn(r'cos[0-9]\(2b' + str(subscript) + '\)', increment_power, subterm)
                            if n == 0:
                                subterm = subterm + 'cos1(2b' + str(subscript) + ')'
                            else:
                                subterm = new_power
                            if new_term_cos == '':
                                new_term_cos = subterm
                            else:
                                new_term_cos = new_term_cos + '+' + subterm
                        if new_terms[term] == '':
                            new_terms[term] = new_term_cos
                        else:
                            new_terms[term] = new_terms[term] + '+' + new_term_cos
                        new_term_sin = ''
                        coeff, pauli = multiply_pauli(term[node], 'x')
                        new_term = term[:node] + pauli + term[node + 1:]
                        for subterm in terms[term].split('+'):
                            new_power, n = re.subn(r'sin[0-9]\(2b' + str(subscript) + '\)', increment_power, subterm)
                            if n == 0:
                                subterm = subterm + fmt_coeff(-1j * coeff) + 'sin1(2b' + str(subscript) + ')'
                            else:
                                subterm = fmt_coeff(-1j * coeff) + new_power
                            if new_term_sin == '':
                                new_term_sin = subterm
                            else:
                                new_term_sin = new_term_sin + '+' + subterm

                        if new_term in new_terms:
                            if new_terms[new_term] == '':
                                new_terms[new_term] = new_term_sin
                            else:
                                new_terms[new_term] = new_terms[new_term] + '+' + new_term_sin
                        else:
                            new_terms[new_term] = new_term_sin
                    else:
                        new_terms[term] = terms[term]
                terms = new_terms.copy()
        elif p % 2 == 0:
            # Apply cost Hamiltonian
            for edge in graph.edges:
                new_terms = {key: '' for key in terms}

                for term in terms:
                    # If it doesn't commute
                    if not commute_with_cost(term):
                        # Implement cos
                        new_term_cos = ''
                        for subterm in terms[term].split('+'):
                            new_coeff, n = re.subn(r'cos[0-9]\(2g' + str(subscript) + '\)', increment_power, subterm)
                            # Update new_terms
                            if n == 0:
                                subterm = subterm + 'cos1(2g' + str(subscript) + ')'
                            else:
                                subterm = new_coeff
                            if new_term_cos == '':
                                new_term_cos = subterm
                            else:
                                new_term_cos = new_term_cos + '+' + subterm
                        if new_terms[term] == '':
                            new_terms[term] = new_term_cos
                        else:
                            new_terms[term] = new_terms[term] + '+' + new_term_cos
                        new_term_sin = ''
                        coeff0, pauli0 = multiply_pauli(term[edge[0]], 'z')
                        coeff1, pauli1 = multiply_pauli(term[edge[1]], 'z')
                        coeff = coeff0 * coeff1
                        new_term = term[:edge[0]] + pauli0 + term[edge[0] + 1:edge[1]] + pauli1 + term[edge[1] + 1:]
                        for subterm in terms[term].split('+'):
                            new_coeff, n = re.subn(r'sin[0-9]\(2g' + str(subscript) + '\)', increment_power, subterm)
                            if n == 0:
                                subterm = subterm + fmt_coeff(-1j * coeff) + 'sin1(2g' + str(subscript) + ')'
                            else:
                                subterm = fmt_coeff(-1j * coeff) + new_coeff

                            if new_term_sin == '':
                                new_term_sin = subterm
                            else:
                                new_term_sin = new_term_sin + '+' + subterm
                        if new_term in new_terms:
                            if new_terms[new_term] != '':
                                new_terms[new_term] = new_terms[new_term] + '+' + new_term_sin
                            else:
                                new_terms[new_term] = new_term_sin
                        else:
                            new_terms[new_term] = new_term_sin
                    else:
                        new_terms[term] = terms[term]
                terms = new_terms.copy()

    return terms


def format_to_latex(terms, basis='z'):
    def format_term(term):
        formatted_term = ''
        for (_, t) in enumerate(term):
            if t == 'x':
                formatted_term += ('X_'+str(_))
            elif t == 'y':
                formatted_term += ('Y_' + str(_))
            elif t == 'z':
                formatted_term += ('Z_' + str(_))
        return formatted_term

    def format_sine_gamma(matchobj):
        if int(matchobj.group()[3]) == 1:
            return "\\" + matchobj.group()[:3] + '(2\gamma_{' + \
                   matchobj.group()[-2] + '})'
        return "\\" +matchobj.group()[:3] + '^{'+ str(int(matchobj.group()[3]))+'}' + '(2\gamma_{'+matchobj.group()[-2]+'})'

    def format_sine_beta(matchobj):
        if int(matchobj.group()[3]) == 1:
            return "\\" + matchobj.group()[:3] + '(2\\beta_{' + \
                   matchobj.group()[-2] + '})'
        return "\\" + matchobj.group()[:3] + '^{' + str(int(matchobj.group()[3])) + '}' + '(2\\beta_{' + \
               matchobj.group()[-2] + '})'

    def format_cosine_beta(matchobj):
        if int(matchobj.group()[3]) == 1:
            return "\\" + matchobj.group()[:3]+ '(2\\beta_{' + \
                   matchobj.group()[-2] + '})'
        return "\\" + matchobj.group()[:3] + '^{' + str(int(matchobj.group()[3])) + '}' + '(2\\beta_{' + \
               matchobj.group()[-2] + '})'

    def format_cosine_gamma(matchobj):
        if int(matchobj.group()[3]) == 1:
            return "\\" + matchobj.group()[:3] + '(2\gamma_{' + \
                   matchobj.group()[-2] + '})'
        return "\\" +matchobj.group()[:3] + '^{'+ str(int(matchobj.group()[3]))+'}' + '(2\gamma_{'+matchobj.group()[-2]+'})'

    def print_formatted_term(term):
        values = terms[term].split('+')
        for (v, value) in enumerate(values):
            coeff = ''
            if value.count('[-1]') % 2 == 1:
                coeff = '-'
            else:
                if v != 0:
                    coeff = '+'
            value = value.replace('[-1]', '')
            value, n = re.subn(r'sin[0-9]\(2g[0-9]\)', format_sine_gamma, value)
            value, n = re.subn(r'sin[0-9]\(2b[0-9]\)', format_sine_beta, value)
            value, n = re.subn(r'cos[0-9]\(2g[0-9]\)', format_cosine_gamma, value)
            value, n = re.subn(r'cos[0-9]\(2b[0-9]\)', format_cosine_beta, value)
            values[v] = coeff + value

        formatted_value = '&['
        for value in values:
            formatted_value += value
        formatted_value += ']' + format_term(term)
        print(formatted_value + ' \\nonumber \\\\')
        print('+')

    for term in terms:
        if basis == 'z':
            if terms[term] != '' and 'y' not in term and 'x' not in term:
                print_formatted_term(term)

        elif basis == 'x':
            if terms[term] != '' and 'y' not in term and 'z' not in term:
                print_formatted_term(term)



from qsim.graph_algorithms.graph import branching_tree_from_edge

graph = branching_tree_from_edge([2, 2], visualize=False)
terms = heisenberg_operator((0, 1), graph, 6)

format_to_latex(terms, basis='x')



