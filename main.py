from enum import Enum, auto
from sys import argv
class TokenType(Enum):
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    RETURN = auto()
    INT = auto()
    FLOAT = auto()
    CHAR = auto()
    STRING = auto()
    VOID = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MOD = auto()
    ASSIGN = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()
    IDENTIFIER = auto()
    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()
    CHAR_LITERAL = auto()
    STRING_LITERAL = auto()
    EOF = auto()
class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    def __repr__(self):
        return f'Token({self.type}, {repr(self.value)}, {self.line}, {self.column})'
class LexicalError(Exception):
    def __init__(self, message, line, column):
        super().__init__(f'Lexical error at {line}:{column}: {message}')
        self.line = line
        self.column = column
class Lexer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        self.current_char = ''
        self.position = -1
        self.line = 1
        self.column = 0
        self.advance()
    def advance(self):
        self.position += 1
        if self.position < len(self.source_code):
            self.current_char = self.source_code[self.position]
            self.column += 1
        else:
            self.current_char = None
        if self.current_char == '\n':
            self.line += 1
            self.column = 0
    def add_token(self, type, value=''):
        self.tokens.append(Token(type, value, self.line, self.column))
    def tokenize(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.advance()
            elif self.current_char.isalpha() or self.current_char == '_':
                self.tokenize_identifier()
            elif self.current_char.isdigit():
                self.tokenize_number()
            elif self.current_char == '+':
                self.add_token(TokenType.PLUS)
                self.advance()
            elif self.current_char == '-':
                self.add_token(TokenType.MINUS)
                self.advance()
            elif self.current_char == '*':
                self.add_token(TokenType.MULTIPLY)
                self.advance()
            elif self.current_char == '/':
                self.add_token(TokenType.DIVIDE)
                self.advance()
            elif self.current_char == '%':
                self.add_token(TokenType.MOD)
                self.advance()
            elif self.current_char == '=':
                self.tokenize_operator('=', TokenType.ASSIGN, TokenType.EQUAL)
            elif self.current_char == '!':
                self.tokenize_operator('!', None, TokenType.NOT_EQUAL)
            elif self.current_char == '<':
                self.tokenize_operator('<', TokenType.LESS_THAN, TokenType.LESS_EQUAL)
            elif self.current_char == '>':
                self.tokenize_operator('>', TokenType.GREATER_THAN, TokenType.GREATER_EQUAL)
            elif self.current_char == '(':
                self.add_token(TokenType.LEFT_PAREN)
                self.advance()
            elif self.current_char == ')':
                self.add_token(TokenType.RIGHT_PAREN)
                self.advance()
            elif self.current_char == '{':
                self.add_token(TokenType.LEFT_BRACE)
                self.advance()
            elif self.current_char == '}':
                self.add_token(TokenType.RIGHT_BRACE)
                self.advance()
            elif self.current_char == ';':
                self.add_token(TokenType.SEMICOLON)
                self.advance()
            elif self.current_char == ',':
                self.add_token(TokenType.COMMA)
                self.advance()
            elif self.current_char == '\'' or self.current_char == '"':
                self.tokenize_string_literal()
            else:
                raise LexicalError(f"Unknown character '{self.current_char}'", self.line, self.column)
        self.add_token(TokenType.EOF)
        return self.tokens
    def tokenize_identifier(self):
        start_pos = self.position
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            self.advance()
        value = self.source_code[start_pos:self.position]
        token_type = self.keyword_or_identifier(value)
        self.add_token(token_type, value)
    def keyword_or_identifier(self, value):
        keywords = {
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'return': TokenType.RETURN,
            'int': TokenType.INT,
            'float': TokenType.FLOAT,
            'char': TokenType.CHAR,
            'void': TokenType.VOID
        }
        return keywords.get(value, TokenType.IDENTIFIER)
    def tokenize_number(self):
        start_pos = self.position
        while self.current_char is not None and self.current_char.isdigit():
            self.advance()
        if self.current_char == '.':
            self.advance()
            if not self.current_char.isdigit():
                raise LexicalError("Malformed float literal", self.line, self.column)
            while self.current_char is not None and self.current_char.isdigit():
                self.advance()
            value = self.source_code[start_pos:self.position]
            self.add_token(TokenType.FLOAT_LITERAL, value)
        else:
            value = self.source_code[start_pos:self.position]
            self.add_token(TokenType.INTEGER_LITERAL, value)
    def tokenize_operator(self, char, single_token, double_token):
        self.advance()
        if self.current_char == '=':
            self.add_token(double_token)
            self.advance()
        elif single_token is not None:
            self.add_token(single_token)
    def tokenize_char_literal(self):
        self.advance()  
        if self.current_char is None or self.current_char == '\'':
            raise LexicalError("Empty character literal", self.line, self.column)
        char_value = self.current_char
        self.advance()
        if self.current_char != '\'':
            raise LexicalError("Unterminated character literal", self.line, self.column)
        self.advance()
        self.add_token(TokenType.CHAR_LITERAL, char_value)
    def tokenize_string_literal(self):
        self.advance()  
        string_value = ''
        while self.current_char is not None and self.current_char != '"' and self.current_char != '\'':
            if self.current_char == '\\':  
                self.advance()
                if self.current_char == 'n':
                    string_value += '\n'
                elif self.current_char == 't':
                    string_value += '\t'
                else:
                    string_value += self.current_char
            else:
                string_value += self.current_char
            self.advance()
        if self.current_char is None:
            raise LexicalError("Unterminated string literal", self.line, self.column)
        self.advance()  
        self.add_token(TokenType.STRING_LITERAL, string_value)
class ASTNode:
    pass
class Program(ASTNode):
    def __init__(self, declarations: list):
        self.declarations = declarations
class VariableDeclaration(ASTNode):
    def __init__(self, var_type, name, initializer):
        self.var_type = var_type
        self.name = name
        self.initializer = initializer
class VariableAssignment(ASTNode):
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression
class Parameter(ASTNode):
    def __init__(self, param_type, name):
        self.param_type = param_type
        self.name = name
class Block(ASTNode):
    def __init__(self, statements):
        self.statements = statements
class FunctionDeclaration(ASTNode):
    def __init__(self, return_type, name:str, parameters: list[Parameter], body:Block):
        self.return_type = return_type
        self.name = name
        self.parameters = parameters
        self.body = body
class Argument(ASTNode):
    def __init__(self, expression):
        self.expression = expression
class FunctionCall(ASTNode):
    def __init__(self, name:str, arguments: list[Argument]):
        self.name = name
        self.arguments = arguments
class ExpressionStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression
class IfStatement(ASTNode):
    def __init__(self, condition, then_branch, else_branch):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch
class WhileStatement(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body
class ForStatement(ASTNode):
    def __init__(self, initializer, condition, increment, body):
        self.initializer = initializer
        self.condition = condition
        self.increment = increment
        self.body = body
class ReturnStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression
class BinaryExpression(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right
class UnaryExpression(ASTNode):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand
class Literal(ASTNode):
    def __init__(self, value):
        self.value = value
class Identifier(ASTNode):
    def __init__(self, name):
        self.name = name
class SyntaxError(Exception):
    def __init__(self, message, line, column):
        super().__init__(f'Syntax error at {line}:{column}: {message}')
        self.line = line
        self.column = column
class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens: list[Token] = tokens
        self.current_token_index = 0
        self.current_token: Token = self.tokens[0]
    def advance(self):
        self.current_token_index += 1
        if self.current_token_index < len(self.tokens):
            self.current_token = self.tokens[self.current_token_index]
    def consume(self, token_type):
        if self.current_token.type == token_type:
            self.advance()
        else:
            raise SyntaxError(f"Expected token {token_type}, got {self.current_token.type}", 
                              self.current_token.line, self.current_token.column)
    def parse(self):
        declarations = []
        while self.current_token.type != TokenType.EOF:
            declarations.append(self.parse_declaration())
        return Program(declarations)
    def parse_declaration(self):
        if self.current_token.type in {TokenType.INT, TokenType.FLOAT, TokenType.CHAR, TokenType.VOID}:
            return self.parse_function_or_variable_declaration()
        elif self.current_token.type == TokenType.IDENTIFIER:
            return self.parse_identifier_statement()
        else:
            raise SyntaxError(f"Unexpected token {self.current_token.type}", 
                              self.current_token.line, self.current_token.column)
    def parse_function_or_variable_declaration(self):
        var_type = self.current_token.type
        self.advance()
        name = self.current_token.value
        self.consume(TokenType.IDENTIFIER)
        if self.current_token.type == TokenType.LEFT_PAREN:
            return self.parse_function_declaration(var_type, name)
        else:
            return self.parse_variable_declaration()
    def parse_function_declaration(self, return_type, name):
        self.consume(TokenType.LEFT_PAREN)
        parameters = self.parse_parameters()
        self.consume(TokenType.RIGHT_PAREN)
        body = self.parse_block()
        return FunctionDeclaration(return_type, name, parameters, body)
    def parse_parameters(self):
        parameters = []
        if self.current_token.type != TokenType.RIGHT_PAREN:
            param_type = self.current_token.type
            self.advance()
            param_name = self.current_token.value
            self.consume(TokenType.IDENTIFIER)
            parameters.append(Parameter(param_type, param_name))
            while self.current_token.type == TokenType.COMMA:
                self.advance()
                param_type = self.current_token.type
                self.advance()
                param_name = self.current_token.value
                self.consume(TokenType.IDENTIFIER)
                parameters.append(Parameter(param_type, param_name))
        return parameters
    def parse_variable_declaration(self):
        var_type = self.current_token.type
        self.advance()
        name = self.current_token.value
        self.advance()
        initializer = None
        if self.current_token.type == TokenType.ASSIGN:
            self.advance()
            initializer = self.parse_expression()
        self.consume(TokenType.SEMICOLON)
        return VariableDeclaration(var_type, name, initializer)
    def parse_identifier_statement(self):
        
        
        
        
        name = self.current_token.value
        self.advance()
        if self.current_token.type == TokenType.LEFT_PAREN:
            arguments = self.parse_arguments()
            self.consume(TokenType.SEMICOLON)
            return FunctionCall(name, arguments)
        elif self.current_token.type == TokenType.ASSIGN:
            self.advance()
            expression = self.parse_expression()
            self.consume(TokenType.SEMICOLON)
            return VariableAssignment(name, expression)
        elif self.current_token.type in {TokenType.PLUS, TokenType.MINUS}:
            
            operator = self.current_token.type
            self.advance()
            operand = Identifier(name)
            self.advance()
            self.consume(TokenType.SEMICOLON)
            return UnaryExpression(operator, operand)
        else:
            raise SyntaxError(f"Unexpected token {self.current_token.type}", 
                              self.current_token.line, self.current_token.column)
    def parse_arguments(self):
        self.consume(TokenType.LEFT_PAREN)
        arguments = []
        if self.current_token.type != TokenType.RIGHT_PAREN:
            arguments.append(Argument(self.parse_expression()))
            while self.current_token.type == TokenType.COMMA:
                self.advance()
                arguments.append(Argument(self.parse_expression()))
        self.consume(TokenType.RIGHT_PAREN)
        return arguments
    def parse_block(self):
        self.consume(TokenType.LEFT_BRACE)
        statements = []
        while self.current_token.type != TokenType.RIGHT_BRACE:
            statements.append(self.parse_statement())
        self.consume(TokenType.RIGHT_BRACE)
        return Block(statements)
    def parse_statement(self):
        if self.current_token.type == TokenType.IF:
            return self.parse_if_statement()
        elif self.current_token.type == TokenType.WHILE:
            return self.parse_while_statement()
        elif self.current_token.type == TokenType.FOR:
            return self.parse_for_statement()
        elif self.current_token.type == TokenType.RETURN:
            return self.parse_return_statement()
        elif self.current_token.type == TokenType.LEFT_BRACE:
            return self.parse_block()
        elif self.current_token.type in {TokenType.INT, TokenType.FLOAT, TokenType.CHAR}:
            return self.parse_variable_declaration()
        if self.current_token.type == TokenType.IDENTIFIER:
            return self.parse_identifier_statement()
        else:
            return self.parse_expression_statement()
    def parse_if_statement(self):
        self.consume(TokenType.IF)
        self.consume(TokenType.LEFT_PAREN)
        condition = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN)
        then_branch = self.parse_statement()
        else_branch = None
        if self.current_token.type == TokenType.ELSE:
            self.advance()
            else_branch = self.parse_statement()
        return IfStatement(condition, then_branch, else_branch)
    def parse_while_statement(self):
        self.consume(TokenType.WHILE)
        self.consume(TokenType.LEFT_PAREN)
        condition = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN)
        body = self.parse_statement()
        return WhileStatement(condition, body)
    def parse_for_statement(self):
        self.consume(TokenType.FOR)
        self.consume(TokenType.LEFT_PAREN)
        initializer = self.parse_expression_statement()
        condition = self.parse_expression()
        self.consume(TokenType.SEMICOLON)
        increment = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN)
        body = self.parse_statement()
        return ForStatement(initializer, condition, increment, body)
    def parse_return_statement(self):
        self.consume(TokenType.RETURN)
        expression = self.parse_expression()
        self.consume(TokenType.SEMICOLON)
        return ReturnStatement(expression)
    def parse_expression_statement(self):
        expression = self.parse_expression()
        self.consume(TokenType.SEMICOLON)
        return ExpressionStatement(expression)
    def parse_expression(self):
        return self.parse_equality()
    def parse_equality(self):
        left = self.parse_comparison()
        while self.current_token.type in {TokenType.EQUAL, TokenType.NOT_EQUAL}:
            operator = self.current_token.type
            self.advance()
            right = self.parse_comparison()
            left = BinaryExpression(left, operator, right)
        return left
    def parse_comparison(self):
        left = self.parse_term()
        while self.current_token.type in {TokenType.LESS_THAN, TokenType.GREATER_THAN, TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL}:
            operator = self.current_token.type
            self.advance()
            right = self.parse_term()
            left = BinaryExpression(left, operator, right)
        return left
    def parse_term(self):
        left = self.parse_factor()
        while self.current_token.type in {TokenType.PLUS, TokenType.MINUS}:
            operator = self.current_token.type
            self.advance()
            right = self.parse_factor()
            left = BinaryExpression(left, operator, right)
        return left
    def parse_factor(self):
        left = self.parse_unary()
        while self.current_token.type in {TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MOD}:
            operator = self.current_token.type
            self.advance()
            right = self.parse_unary()
            left = BinaryExpression(left, operator, right)
        return left
    def parse_unary(self):
        if self.current_token.type in {TokenType.PLUS, TokenType.MINUS}:
            operator = self.current_token.type
            self.advance()
            operand = self.parse_primary()
            return UnaryExpression(operator, operand)
        return self.parse_primary()
    def parse_primary(self):
        if self.current_token.type == TokenType.INTEGER_LITERAL:
            value = int(self.current_token.value)
            self.advance()
            return Literal(value)
        elif self.current_token.type == TokenType.FLOAT_LITERAL:
            value = float(self.current_token.value)
            self.advance()
            return Literal(value)
        elif self.current_token.type == TokenType.STRING_LITERAL:
            value = self.current_token.value
            self.advance()
            return Literal(value)
        elif self.current_token.type == TokenType.IDENTIFIER:
            name = self.current_token.value
            self.advance()
            return Identifier(name)
        elif self.current_token.type == TokenType.LEFT_PAREN:
            self.advance()
            expression = self.parse_expression()
            self.consume(TokenType.RIGHT_PAREN)
            return expression
        else:
            raise SyntaxError(f"Unexpected token {self.current_token.type}", 
                              self.current_token.line, self.current_token.column)
class SemanticError(Exception):
    def __init__(self, message, line, column):
        super().__init__(f'Semantic error at {line}:{column}: {message}')
        self.line = line
        self.column = column
class SymbolTable:
    def __init__(self):
        self.symbols = {}
    def declare(self, name, type, line, column):
        if name in self.symbols:
            raise SemanticError(f"Variable '{name}' already declared", line, column)
        self.symbols[name] = type
    def lookup(self, name, line, column):
        if name not in self.symbols:
            raise SemanticError(f"Variable '{name}' not declared", line, column)
        return self.symbols[name]
class SemanticAnalyzer:
    def __init__(self, ast: ASTNode):
        self.ast: ASTNode = ast
        self.symbol_table:SymbolTable = SymbolTable()
    def analyze(self):
        self.visit(self.ast)
    def visit(self, node: ASTNode):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    def generic_visit(self, node: ASTNode):
        raise Exception(f'No visit_{type(node).__name__} method')
    def visit_Program(self, node: Program):
        for declaration in node.declarations:
            self.visit(declaration)
    def visit_FunctionDeclaration(self, node: FunctionDeclaration):
        self.symbol_table.declare(node.name, node.return_type, 0, 0)
        old_symbols = self.symbol_table.symbols.copy()
        for param in node.parameters:
            self.symbol_table.declare(param.name, param.param_type, 0, 0)
        self.visit(node.body)
        self.symbol_table.symbols = old_symbols
    def visit_FunctionCall(self, node: FunctionCall):
        
        if node.name in ['printf', 'scanf']:
            return
        if node.name not in self.symbol_table.symbols:
            raise SemanticError(f"Function '{node.name}' not declared", 0, 0)
        for argument in node.arguments:
            self.visit(argument)
    def visit_Parameter(self, node: Parameter):
        self.symbol_table.declare(node.name, node.param_type, 0, 0)
    def visit_Argument(self, node: Argument):
        self.visit(node.expression)
    def visit_VariableDeclaration(self, node: VariableDeclaration):
        self.symbol_table.declare(node.name, node.var_type, 0, 0)
        if node.initializer is not None:
            self.visit(node.initializer)
    def visit_VariableAssignment(self, node: VariableAssignment):
        var_type = self.symbol_table.lookup(node.name, 0, 0)
        expr_type = self.visit(node.expression)
        if var_type != expr_type:
            raise SemanticError(f"Type mismatch in assignment to '{node.name}'", 0, 0)
    def visit_Block(self, node: Block):
        for statement in node.statements:
            self.visit(statement)
    def visit_IfStatement(self, node: IfStatement):
        self.visit(node.condition)
        self.visit(node.then_branch)
        if node.else_branch is not None:
            self.visit(node.else_branch)
    def visit_WhileStatement(self, node: WhileStatement):
        self.visit(node.condition)
        self.visit(node.body)
    def visit_ForStatement(self, node: ForStatement):
        self.visit(node.initializer)
        self.visit(node.condition)
        self.visit(node.increment)
        self.visit(node.body)
    def visit_ReturnStatement(self, node: ReturnStatement):
        if node.expression is not None:
            self.visit(node.expression)
    def visit_BinaryExpression(self, node: BinaryExpression):
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        if left_type == TokenType.FLOAT or right_type == TokenType.FLOAT:
            return TokenType.FLOAT
        elif left_type == TokenType.CHAR or right_type == TokenType.CHAR:
            return TokenType.CHAR
        elif left_type == TokenType.INT and right_type == TokenType.INT:
            return TokenType.INT
        else:
            raise SemanticError("Unknown type", 0, 0)
    def visit_UnaryExpression(self, node):
        return self.visit(node.operand)
    def visit_Literal(self, node: Literal):
        if isinstance(node.value, int):
            return TokenType.INT
        elif isinstance(node.value, float):
            return TokenType.FLOAT
        elif isinstance(node.value, str):
            return TokenType.CHAR
        else:
            raise SemanticError("Unknown literal type", 0, 0)
    def visit_Identifier(self, node: Identifier):
        return self.symbol_table.lookup(node.name, 0, 0)
    def get_symbol_table(self):
        return self.symbol_table
class ASTToPython:
    def __init__(self, ast):
        self.ast = ast
        self.global_variables = {
            'printf': 'print',
            'scanf':  'input'
        }
    def convert(self):
        return self.visit(self.ast)
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    def generic_visit(self, node):
        raise Exception(f'No visit_{type(node).__name__} method')
    def visit_Program(self, node):
        return '\n'.join(self.visit(declaration) for declaration in node.declarations)
    def visit_FunctionDeclaration(self, node):
        params = ', '.join(self.visit(param) for param in node.parameters)
        body = self.visit(node.body)
        return f"def {node.name}({params}):\n{self.indent(body)}"
    def visit_FunctionCall(self, node: FunctionCall):
        arguments = ', '.join(self.visit(argument) for argument in node.arguments)
        if node.name in self.global_variables:
            return f"{self.global_variables[node.name]}({arguments})"
        return f"{node.name}({arguments})"
    def visit_VariableDeclaration(self, node):
        initializer = self.visit(node.initializer) if node.initializer else "None"
        return f"{node.name} = {initializer}"
    def visit_VariableAssignment(self, node):
        expression = self.visit(node.expression)
        return f"{node.name} = {expression}"
    def visit_Parameter(self, node):
        return node.name
    def visit_Block(self, node):
        return '\n'.join(self.visit(statement) for statement in node.statements)
    def visit_ExpressionStatement(self, node):
        return self.visit(node.expression)
    def visit_IfStatement(self, node):
        condition = self.visit(node.condition)
        then_branch = self.indent(self.visit(node.then_branch))
        else_branch = f"\nelse:\n{self.indent(self.visit(node.else_branch))}" if node.else_branch else ""
        return f"if {condition}:\n{then_branch}{else_branch}"
    def visit_WhileStatement(self, node):
        condition = self.visit(node.condition)
        body = self.indent(self.visit(node.body))
        return f"while {condition}:\n{body}"
    def visit_ForStatement(self, node):
        initializer = self.visit(node.initializer)
        condition = self.visit(node.condition)
        increment = self.visit(node.increment)
        body = self.indent(self.visit(node.body))
        return f"{initializer}\nwhile {condition}:\n{body}\n{increment}"
    def visit_ReturnStatement(self, node):
        expression = self.visit(node.expression) if node.expression else ""
        return f"return {expression}"
    def visit_BinaryExpression(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = self.convert_operator(node.operator)
        return f"{left} {operator} {right}"
    def visit_UnaryExpression(self, node: UnaryExpression):
        operand = self.visit(node.operand)
        operator = '+' if node.operator == TokenType.PLUS else '-'
        return f"{operand}{operator}1"
    def visit_Literal(self, node):
        return repr(node.value)
    def visit_Identifier(self, node:Identifier):
        return node.name
    def visit_Argument(self, node:Argument):
        return self.visit(node.expression)
    def convert_operator(self, operator):
        operators = {
            TokenType.PLUS: "+",
            TokenType.MINUS: "-",
            TokenType.MULTIPLY: "*",
            TokenType.DIVIDE: "/",
            TokenType.MOD: "%",
            TokenType.ASSIGN: "=",
            TokenType.EQUAL: "==",
            TokenType.NOT_EQUAL: "!=",
            TokenType.LESS_THAN: "<",
            TokenType.GREATER_THAN: ">",
            TokenType.LESS_EQUAL: "<=",
            TokenType.GREATER_EQUAL: ">="
        }
        return operators[operator]
    def indent(self, code, level=1):
        return '\n'.join('    ' * level + line for line in code.split('\n'))
def print_ast (node, level=0):
    print('  ' * level + type(node).__name__)
    for child in node.__dict__.values():
        if isinstance(child, list):
            for c in child:
                print_ast(c, level + 1)
        elif isinstance(child, ASTNode):
            print_ast(child, level + 1)
def main():
    filename = argv[1]
    with open(filename) as file:
        source_code = file.read()
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()

    print(
        '#####################################################' + '\n' +
        'Tokens:' + '\n' +
        '#####################################################' + '\n'
    )
    for token in tokens:
        print(f'{token.type.name}: {token.value}')

    parser = Parser(tokens)
    ast = parser.parse()
    
    print(
        '#####################################################' + '\n' +
        'Arbol de sintaxis asbtracta:' + '\n' +
        '#####################################################' + '\n'
    )
    
    print_ast(ast)
    semantic_analyzer = SemanticAnalyzer(ast)
    semantic_analyzer.analyze()
    # impresion de la tabla de simbolos
    symbol_table = semantic_analyzer.get_symbol_table()

    print(
        '#####################################################' + '\n' +
        'Tabla de simbolos:' + '\n' +
        '#####################################################' + '\n'
    )

    for name, type in symbol_table.symbols.items():
        print(f'{name}: {type.name}')
    py_converter = ASTToPython(ast)
    python_code = py_converter.convert()
    
    print(
        '#####################################################' + '\n' +
        'Codigo compilado:' + '\n' +
        '#####################################################' + '\n'
    )
    print(python_code)
    
    print(
        '#####################################################' + '\n' +
        'Ejecucion del codigo:' + '\n' +
        '#####################################################' + '\n'
    )
    exec(python_code)
if __name__ == '__main__':
    main()
