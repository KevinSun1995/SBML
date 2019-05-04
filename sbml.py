import sys
import ply.yacc as yacc
import ply.lex as lex


block_stack = []
fun_id = {}
id = {}

class Node:
    def __init__(self):
        print("init node")

    def evaluate(self):
        return 0

    def execute(self):
        return 0


class SemanticError(Exception):
    def __init__(self, message):
        self.message = message


class NumberNode(Node):
    def __init__(self, v):
        if ('.' in v):
            self.value = float(v)
        else:
            self.value = int(v)

    def evaluate(self):
        return self.value


class IDNode(Node):
    def __init__(self, v):
        self.v = v

    def evaluate(self):
        while isinstance(self.v, Node):
            self.v = self.v.evaluate()
        return id[self.v]


class StatementNode(Node):
    def __init__(self, s):
        self.s = s

    def evaluate(self):
        if self.s:
            return self.s.evaluate();


class AssignNode(Node):
    def __init__(self, var, exp):
        self.var = var
        self.exp = exp

    def evaluate(self):
        x = self.exp
        while isinstance(x, Node):
            x = x.evaluate()
        id[self.var] = x


class FuncNode(Node):
    def __init__(self, id, paras, block, exp):
        self.id = id
        self.paras = paras
        self.block = block
        self.exp = exp

    def evaluate(self):
        fun_id[self.id] = self


class FunCalledNode(Node):
    def __init__(self, f_id, args):
        self.f_id = f_id
        self.args = args

    def evaluate(self):
        x = self.f_id
        func = fun_id[x]
        f_paras = func.paras
        f_block = func.block
        f_exp = func.exp
        if len(f_paras) != len(self.args):
            raise SemanticError('Semantic Error')
        args_dic = {}
        for i in range(len(self.args)):
            x = f_paras[i].v
            y = self.args[i]
            while isinstance(y, Node):
                y = y.evaluate()
            args_dic[x] = y
        temp = {}
        for key, value in id.items():
            temp[key] = value
        for key, value in args_dic.items():
            id[key] = value
        f_block.evaluate()
        result = id[f_exp.v]
        id.clear()
        for key, value in temp.items():
            id[key] = value
        return result


class ListAssignNode(Node):
    def __init__(self, var, index, exp):
        self.var = var
        self.index = index
        self.exp = exp

    def evaluate(self):
        x = self.exp
        while isinstance(x, Node):
            x = x.evaluate()
        y = self.index
        while isinstance(y, Node):
            y = y.evaluate()
        if not isinstance(id[self.var], list):
            raise SemanticError('Semantic Error')
        id[y] = x


class ListDoubleAssignNode(Node):
    def __init__(self, var, index1, index2, exp):
        self.var = var
        self.index1 = index1
        self.index2 = index2
        self.exp = exp

    def evaluate(self):
        x = self.exp
        while isinstance(x, Node):
            x = x.evaluate()
        y = self.index1
        while isinstance(y, Node):
            y = y.evaluate()
        z = self.index2
        while isinstance(z, Node):
            z = z.evaluate()
        if not isinstance(id[self.var], list):
            raise SemanticError('Semantic Error')
        if not isinstance(id[self.var][y], list):
            raise SemanticError('Semantic Error')
        id[y][z] = x


class PrintNode(Node):
    def __init__(self, v):
        self.value = v

    def evaluate(self):
        x = self.value
        while isinstance(x, Node):
            x = x.evaluate()
        if isinstance(x, str):
            print(x[1:-1])
        else:
            print(x)


class IfNode(Node):
    def __init__(self, exp, block):
        self.exp = exp
        self.block = block

    def evaluate(self):
        x= self.exp
        while isinstance(x, Node):
            x = x.evaluate()
        if not isinstance(x, bool):
            raise SemanticError('Semantic Error')
        else:
            if x:
                self.block.evaluate()


class IfElseNode(Node):
    def __init__(self, exp, if_block, else_block):
        self.exp =exp
        self.if_block = if_block
        self.else_block = else_block

    def evaluate(self):
        x = self.exp
        while isinstance(x, Node):
            x = x.evaluate()
        if not isinstance(x, bool):
            raise SemanticError('Semantic Error')
        else:
            if x:
                self.if_block.evaluate()
            else:
                self.else_block.evaluate()


class WhileNode(Node):
    def __init__(self, exp, block):
        self.exp = exp
        self.block = block

    def evaluate(self):
        while True:
            temp = self.exp
            while isinstance(temp, Node):
                temp = temp.evaluate()
            if not isinstance(temp, bool):
                raise SemanticError('Semantic Error')
            if temp:
                self.block.evaluate()
            else:
                break


class BooleanNode(Node):
    def __init__(self, v):
        if v == 'True':
            self.value = True
        else:
            self.value = False

    def evaluate(self):
        return self.value


class StringNode(Node):
    def __init__(self, v):
        self.value = v

    def evaluate(self):
        return self.value


class ListNode(Node):
    def __init__(self, v):
        self.value = v

    def evaluate(self):
        x = []
        for n in self.value:
            while isinstance(n, Node):
                n = n.evaluate()
            if isinstance(n, str):
                x.append(n[1:-1])
            else:
                x.append(n)
        return x


class TupleNode(Node):
    def __init__(self, v):
        self.value = v

    def evaluate(self):
        return self.value


class TupleIndex(Node):
    def __init__(self, index, tuples):
        self.index = index
        self.tuples = tuples

    def evaluate(self):
        x = self.index
        while isinstance(x, Node):
            x = x.evaluate()
        y = self.tuples
        while isinstance(y, Node):
            y = y.evaluate()
        if x > len(y):
            raise SemanticError('index out of tuple range')
        return y[x]


class Index(Node):
    def __init__(self, lists, index):
        self.index = index
        self.lists = lists

    def evaluate(self):
        x = self.lists
        while isinstance(x, Node):
            x = x.evaluate()
        y = self.index
        while isinstance(y, Node):
            y = y.evaluate()
        if isinstance(x, str):
            if y > len(x)-2:
                raise SemanticError('index out of string range')
            else:
                result = x[y+1]
                return '\'' + result + '\''
        else:
            if y > len(x):
                raise SemanticError('index out of list range')
            else:
                result = x[y]
                return result


class NotNode(Node):
    def __init__(self, op, v1):
        self.v1 = v1
        self.op = op

    def evaluate(self):
        if not isinstance(self.v1.evaluate(), bool):
            raise SemanticError("SEMANTIC ERROR")
        else:
            return not self.v1.evaluate()


class BopNode(Node):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def is_int(self, x, y):
        if isinstance(x, int) and isinstance(y, int):
            return True
        else:
            raise SemanticError("SEMANTIC ERROR")

    def is_int_or_real(self, x, y):
        if (not isinstance(x, int) and not isinstance(x, float)) or (not isinstance(y, int) and not isinstance(y, float)):
            raise SemanticError("SEMANTIC ERROR")
        else:
            return True

    def is_number_or_string(self, x, y):
        if (isinstance(x, int) or isinstance(x, float)) and (isinstance(y, int) or isinstance(y, float)):
            return True
        elif isinstance(x, str) and isinstance(y, str):
            return True
        else:
            raise SemanticError("SEMANTIC ERROR")

    def is_number_or_string_or_list(self, x, y):
        if (isinstance(x, int) or isinstance(x, float))and (isinstance(y, int) or isinstance(y, float)):
            return True
        elif isinstance(x, str) and isinstance(y, str):
            return True
        elif isinstance(x, list) and isinstance(y, list):
            return True
        else:
            raise SemanticError("SEMANTIC ERROR")

    def evaluate(self):
        if self.op == '+':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_number_or_string_or_list(x, y)
            if isinstance(x, str):
                return x[:-1] + y[1:]
            return x + y
        elif self.op == '-':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_number_or_string_or_list(x, y)
            if isinstance(x, list):
                for i in y:
                    if i in x:
                        x.remove(i)
                    else:
                        raise SemanticError('SEMANTIC ERROR')
                return x
            else:
                return NumberNode(str(x - y))
        elif self.op == '*':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_int_or_real(x, y)
            return NumberNode(str(x * y))
        elif self.op == '/':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_int_or_real(x, y)
            return NumberNode(str(x / y))
        elif self.op == '**':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_int_or_real(x, y)
            return NumberNode(str(x ** y))
        elif self.op == 'mod':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_int(x, y)
            return NumberNode(str(x % y))
        elif self.op == 'div':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_int(x, y)
            return NumberNode(str(x // y))
        elif self.op == 'in':
            x = self.v1
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2
            while isinstance(y, Node):
                y = y.evaluate()
            if isinstance(y, list):
                return x in y
            elif isinstance(y, str) and isinstance(x, str):
                return x[1:-1] in y()[1:-1]
            else:
                raise SemanticError("SEMANTIC ERROR")
        elif self.op == '::':
            x = self.v1
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2
            while isinstance(y, Node):
                y = y.evaluate()
            if isinstance(y, list):
                return [x] + y
            elif isinstance(y, str):
                return x[:-1] + y[1:]
            else:
                raise SemanticError("SEMANTIC ERROR")
        elif self.op == '<':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_number_or_string(x, y)
            return x < y
        elif self.op == '>':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_number_or_string(x, y)
            return x > y
        elif self.op == '==':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_number_or_string(x, y)
            return x == y
        elif self.op == '<>':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_number_or_string(x, y)
            return x != y
        elif self.op == '<=':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_number_or_string(x, y)
            return x <= y
        elif self.op == '>=':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            self.is_number_or_string(x, y)
            return x >= y
        elif self.op == 'andalso':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            if not isinstance(x, bool) or not isinstance(y, bool):
                raise SemanticError("SEMANTIC ERROR")
            else:
                return x and y
        elif self.op == 'orelse':
            x = self.v1.evaluate()
            while isinstance(x, Node):
                x = x.evaluate()
            y = self.v2.evaluate()
            while isinstance(y, Node):
                y = y.evaluate()
            if not isinstance(x, bool) or not isinstance(y, bool):
                raise SemanticError("SEMANTIC ERROR")
            else:
                return x or y


class ProgramNode(Node):
    def __init__(self, funcs, block):
        self.funcs = funcs
        self.block = block

    def evaluate(self):
        for f in self.funcs:
            f.evaluate()
        self.block.evaluate()


class BlockNode(Node):
    def __init__(self, sl):
        self.statementList = sl

    def evaluate(self):
        for s in self.statementList:
            s.evaluate()


tokens = [
    'LPAREN', 'RPAREN', 'ID',
    'NUMBER', 'STRING',
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'EXP',
    'CONC', 'LESS', 'MORE', 'EQUAL', 'NOTEQUAL', 'LESSE', 'MOREE',
    'EQUALS',
    'SEMI', 'COMMA', 'POUND',
    'LBRACKET', 'RBRACKET',
    'L_CURLY', 'R_CURLY'

]


reserved = {
    'while': 'WHILE',
    'if': 'IF',
    'print': 'PRINT',
    'mod': 'MOD',
    'div': 'DIV',
    'else': 'ELSE',
    'andalso': 'ANDALSO',
    'orelse':'ORELSE',
    'True': 'TRUE',
    'False': 'FALSE',
    'not' : 'NOT',
    'in': 'IN',
    'fun': 'FUN'
}

tokens += reserved.values()
# Tokens

t_EQUALS = r'='
t_PRINT = r'print'
t_L_CURLY = r'\{'
t_R_CURLY = r'\}'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_MOD = r'mod'
t_DIV = r'div'
t_EXP = r'\*\*'

t_SEMI = r';'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_POUND= r'\#'
t_CONC = r'::'
t_LESS = r'<'
t_MORE = r'>'
t_EQUAL = r'=='
t_NOTEQUAL = r'<>'
t_LESSE = r'<='
t_MOREE = r'>='


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in reserved:
        t.type = reserved[t.value]
    return t


def t_NUMBER(t):
    r'-?\d*(\d\.|\.\d)\d*e-?\d+ | -?\d*(\d\.|\.\d)\d*| \d+'
    try:
        t.value = NumberNode(t.value)
    except ValueError:
        print("Integer value too large %d", t.value)
        t.value = 0
    return t


def t_STRING(t):
    r'\'[^\'\"]*\'|\"[^\'\"]*\"'
    t.value = StringNode('\'' + t.value[1:-1] + '\'')
    return t


# Ignored characters
t_ignore = " \t"


def t_error(t):
    raise SyntaxError('SYNTAX ERROR')


# Build the lexer
lex.lex(debug=0)

# Parsing rules
precedence = (
    ('left', 'ANDALSO', 'ORELSE'),
    ('left', 'NOT'),
    ('left', 'LESS', 'MORE', 'EQUAL', 'NOTEQUAL', 'LESSE', 'MOREE'),
    ('left', 'CONC'),
    ('left', 'IN'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'MOD', 'DIV'),
    ('right', 'EXP'),
    ('left', 'POUND'),
    ('left', 'LPAREN', 'RPAREN')
)

def p_program(p):
    '''
    program : functions block
    '''
    p[0] = ProgramNode(p[1], p[2])


def p_fun(p):
    '''
    function : FUN ID LPAREN series RPAREN EQUALS result
    '''
    for n in p[4]:
        if not isinstance(n, IDNode):
            raise SemanticError('Semantic Error')
    p[0] = FuncNode(p[2], p[4], p[7][0], p[7][1])


def p_func_list_val(p):
    '''
    functions : function
    '''
    p[0] = [p[1]]


def p_func_list(p):
    '''
    functions : functions function
    '''
    p[0] = p[1] + [p[2]]


def p_result(p):
    'result : block expression SEMI'
    p[0] = [p[1], p[2]]


def p_block(p):
    '''
     block : L_CURLY statement_list R_CURLY
            | L_CURLY block R_CURLY
            | L_CURLY R_CURLY
    '''
    if len(p) == 3:
        p[0] = BlockNode([])
    elif isinstance(p[2], BlockNode):
        p[0] = BlockNode([p[2]])
    else:
        p[0] = BlockNode(p[2])


def p_statement_block(p):
    'statement : block'
    p[0] = p[1]


def p_statement_if(p):
    '''
    statement : IF LPAREN expression RPAREN block
    '''

    p[0] = IfNode(p[3], p[5])


def p_statement_if_else(p):
    '''
    statement : IF LPAREN expression RPAREN block ELSE block
    '''
    p[0] = IfElseNode(p[3], p[5], p[7])


def p_statement_while(p):
    '''
    statement : WHILE LPAREN expression RPAREN block
    '''
    p[0] = WhileNode(p[3], p[5])


def p_statement_list(p):
    '''
     statement_list : statement_list statement
    '''
    p[0] = p[1] + [p[2]]


def p_statement_list_val(p):
    '''
    statement_list : statement
    '''
    p[0] = [p[1]]


def p_print_smt(t):
    """
    statement : PRINT LPAREN expression RPAREN SEMI
    """
    t[0] = StatementNode(PrintNode(t[3]))


def p_statement_func_called(t):
    '''
    FUNCALLED : ID LPAREN series RPAREN %prec POUND
    '''
    t[0] = FunCalledNode(t[1], t[3])


def p_statement_assign(t):
    '''
    statement : ID EQUALS expression SEMI
              | ID EQUALS FUNCALLED SEMI
    '''
    t[0] = AssignNode(t[1], t[3])


def p_statement_funcalled(p):
    'statement : FUNCALLED SEMI'
    p[0] = p[1]


def p_statement_list_assign(t):
    'statement : ID LBRACKET expression RBRACKET EQUALS expression SEMI'
    t[0] = ListAssignNode(t[1], t[3], t[6])


def p_statement_list_double_assign(t):
    'statement : ID LBRACKET expression RBRACKET LBRACKET expression RBRACKET EQUALS expression SEMI'
    t[0] = ListDoubleAssignNode(t[1], t[3], t[6], t[9])


def p_expression_paren(t):
    '''
    expression : LPAREN expression RPAREN
    '''

    t[0] = t[2]


def p_expression_id(t):
    'factor : ID'
    try:
        t[0] = IDNode(t[1])
    except LookupError:
        print("Undefined name '%s'" % t[1])
        t[0] = 0


def p_expression_pond(t):
    '''
    expression : POUND expression expression
    '''
    t[0] = TupleIndex(t[2], t[3])


def p_expression_index(t):
    '''
    expression : list LBRACKET factor RBRACKET
               | ID LBRACKET factor RBRACKET
    '''
    if isinstance(t[3], StringNode) or isinstance(t[3], BooleanNode):
        raise SemanticError('Semantic Error')
    if not isinstance(t[1] , list):
        t[1] = IDNode(t[1])
    t[0] = Index(t[1], t[3])


def p_expression_list(t):
    '''
    expression : list
    '''
    t[0] = t[1]


def p_expression_tuple(t):
    '''
    expression : tuple
    '''
    t[0] = t[1]


def p_expression_series(t):
    '''
    series : series COMMA expression
            | expression
    '''
    if len(t) == 2:
        t[0] = [t[1]]
    else:
        if isinstance(t[1], list):
            t[1].append(t[3])
            t[0] = t[1]
        else:
            t[0] = [t[1], t[3]]


def p_expression_create_list(t):
    '''
    list : LBRACKET series RBRACKET
        | LBRACKET RBRACKET
    '''
    if len(t) == 3:
        t[0] = ListNode([])
    else:
        t[0] = []
        for n in t[2]:
            t[0].append(n)
        t[0] = ListNode(t[0])


def p_expression_create_tuple(t):
    '''
    tuple : LPAREN series RPAREN
        | LPAREN RPAREN
    '''
    if len(t) == 3:
        t[0] = TupleNode(())
    else:
        t[0] = []
        for n in t[2]:
            t[0].append(n.value)
        t[0] = TupleNode(tuple(t[0]))


def p_expression_notop(t):
    '''expression : NOT expression'''
    t[0] = NotNode(t[1], t[2])


def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression EXP expression
                  | expression MOD expression
                  | expression DIV expression
                  | expression IN expression
                  | expression CONC expression
                  | expression LESS expression
                  | expression MORE expression
                  | expression EQUAL expression
                  | expression NOTEQUAL expression
                  | expression LESSE expression
                  | expression MOREE expression
                  | expression ANDALSO expression
                  | expression ORELSE expression
    '''
    t[0] = BopNode(t[2], t[1], t[3])


def p_expression_factor(t):
    'expression : factor'
    t[0] = t[1]


def p_expression_number(t):
    'factor : NUMBER'
    t[0] = t[1]


def p_expression_func(t):
    'factor : FUNCALLED'
    t[0] = t[1]


def p_expression_boolean(t):
    '''expression : TRUE
                  | FALSE '''
    t[0] = BooleanNode(t[1])


def p_expression_negative(t):
    'factor : MINUS NUMBER'
    x = t[2].evaluate()
    t[0] = NumberNode('-'+str(x))


def p_expression_string(t):
    'factor : STRING'
    t[0] = t[1]


def p_error(t):
    pass


yacc.yacc(debug=0)


if len(sys.argv) != 2:
    sys.exit("invalid arguments")
fd = open(sys.argv[1], 'r')
code = ""
data = ""
for line in fd:
    code = line.strip()
    if code == '':
        continue
    data += code
try:
    lex.input(data)
    while True:
        token = lex.token()
        if not token:
            break
        # print(token)
    root = yacc.parse(data)
    root.evaluate()
except SemanticError:
    print("SEMANTIC ERROR")
except Exception:
    print("SYNTAX ERROR")
