import os
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI  # Import atualizado
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

def lookup_term(term: str) -> str:
    knowledge_base = {
        "iso27001": "Uma norma internacional de segurança da informação.",
        "autenticação multifator": "Processo de login que requer mais de um fator de confirmação.",
        "usabilidade": "A facilidade com que um usuário pode interagir com um sistema para atingir um objetivo específico.",
        "requisito funcional": "Uma funcionalidade que o sistema deve fornecer.",
        "requisito não funcional": "Uma propriedade ou restrição de qualidade do sistema, como desempenho ou segurança."
    }
    return knowledge_base.get(term.lower(), f"Não encontrei uma definição para o termo '{term}'.")

tools = [
    Tool(
        name="TermLookup",
        func=lookup_term,
        description="Use esta ferramenta para obter definicoes de termos relacionados a engenharia de software. Informe o termo como entrada."
    )
]

openai_api_key = os.getenv("apikey_chatgpt4")

if not openai_api_key:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida. Defina sua chave antes de executar.")

# Use ChatOpenAI para modelos de chat
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=openai_api_key)

system_template = """Você é um agente de IA que atua como um assistente de Engenharia de Requisitos.
Você recebe um texto contendo informações sobre requisitos de um sistema.
Sua tarefa:
1. Identificar requisitos funcionais e não funcionais.
2. Sugerir melhorias.
3. Identificar inconsistências ou lacunas, se houver.

Você pode consultar a ferramenta TermLookup para entender termos específicos.
Ao final, produza uma especificação de requisitos limpa em formato Markdown, dividida em:
- Requisitos Funcionais
- Requisitos Não Funcionais
- Sugestões de Melhoria
- Inconsistências/Lacunas detectadas (se houver)

Se precisar entender melhor um termo, use a ferramenta TermLookup.
"""

human_template = """Texto de entrada (requisitos):

"{requirements}"

Por favor, processe essas informações.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

def process_requirements(requirements_text: str):
    prompt = chat_prompt.format_prompt(requirements=requirements_text)
    # O agent pode ser inicializado usando a mesma função, já que o ChatOpenAI é compatível
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    response = agent.run(prompt.to_messages())
    return response

if __name__ == "__main__":
    example_requirements = """
    O sistema deve permitir que o usuário faça login utilizando credenciais internas da empresa.
    Uma vez autenticado, o usuário pode acessar um relatório diário de vendas.
    O relatório deve ser gerado em menos de 2 segundos.
    O sistema deve estar em conformidade com a norma ISO27001.
    É desejável que o sistema seja fácil de usar (boa usabilidade), mas não foi especificado o tipo de autenticação (simples ou multifator).
    """

    resultado = process_requirements(example_requirements)
    print("\n--- Especificação Gerada ---\n")
    print(resultado)
