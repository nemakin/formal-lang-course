from antlr4 import InputStream, CommonTokenStream, ParserRuleContext, TerminalNode
from project.grammar.GraphLexer import GraphLexer
from project.grammar.GraphParser import GraphParser


def program_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    input_stream = InputStream(program)
    lexer = GraphLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = GraphParser(token_stream)
    tree = parser.prog()

    num_errors = parser.getNumberOfSyntaxErrors()
    if num_errors > 0:
        return None, False
    return tree, True


def nodes_count(tree: ParserRuleContext) -> int:
    count = 1
    for child in tree.children:
        if isinstance(child, ParserRuleContext):
            count += 1

    return count


def tree_to_program(tree: ParserRuleContext) -> str:
    program = ""
    for child in tree.children:
        if isinstance(child, ParserRuleContext):
            program += tree_to_program(child)
        if isinstance(child, TerminalNode):
            program += child.getText() + " "
    return program
