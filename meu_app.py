import streamlit as st
from rag import perguntando
from pathlib import Path

css_path = Path("style.css")
html_path = Path("style.html")

with css_path.open("r", encoding="utf-8") as f:
    css_content = f"<style>{f.read()}</style>"
css_content = css_content.replace("\ufeff", "")

with html_path.open("r", encoding="utf-8") as f:
    html_content = f.read()

def app():
    company_logo = 'https://i.ibb.co/BVGpvMPD/imagem-streamlit.png'
    st.components.v1.html(html_content)

    st.sidebar.header("Informações")
    st.sidebar.text("""
    - Este aplicativo foi desenvolvido com o objetivo de testar e potencializar meus aprendizados em LLMs.\n
    - Para a criação da aplicação, utilizei o framework LangChain para integrar com os sistemas da OpenAI. \n
    - A inclusão das informações pessoais foi realizada por meio de um pré-processamento estruturado:  
        - Leitura de um arquivo contendo um resumo pessoal. 
        - Divisão do conteúdo em chunks de 150 caracteres.
        - Geração de embeddings utilizando o OpenAIEmbeddings. 

    - A RAG foi construída com um template que seleciona os 3 chunks mais relevantes de acordo com a consulta. \n  
    - Caso a similaridade da pergunta seja inferior a 0,4 em relação ao chunk mais relevante, retornamos que não foi possível encontrar uma resposta nos dados disponíveis.
    """)
    st.sidebar.markdown(
        """
        <hr>
        <p style='font-size:12px;color:gray;'>
            © 2025 Alice Ferreira de Souza - Todos os direitos reservados <br>
            <a href="mailto:alicefeerreira490@gmail.com" style="color:gray; text-decoration:none;">alicefeerreira490@gmail.com</a> <br>
            <a href="https://www.linkedin.com/in/alice-ferreira-17b5281a9?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app" 
                target="_blank" style="color:gray; text-decoration:none;">LinkedIn</a>
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown(css_content, unsafe_allow_html=True)

    st.set_page_config(
        page_title="Chatbot da Alice",
        page_icon=company_logo,
        layout="centered"
    )
    st.write("""
        💬 Faça uma pergunta sobre minha experiência profissional e projetos!  
        Você pode perguntar sobre minha trajetória em ciência de dados, projetos de Machine Learning, LLMs, RAGs, ou habilidades técnicas.  

        Por exemplo:  
        💻 "Como a Alice aplica Machine Learning em projetos de personalização de apps?"  
        💡 "Qual foi a experiência dela desenvolvendo chatbots com LLMs e RAGs?"  
        📊 "Quais ferramentas e dashboards a Alice já implementou para análise de dados?"  
        🏐 "Quais hobbies e atividades a Alice gosta de praticar fora do trabalho?"
    """)

    ### etapa de mensagens:
    mensagem_usuario = st.chat_input("Faça sua pergunta sobre a Alice.")

    if mensagem_usuario is not None:

        # preciso salvar os dados de mensagem para ter o istorico salvi
        if "mensagens" in st.session_state:
            mensagens = st.session_state["mensagens"]
        else:
            mensagens = []
            st.session_state["mensagens"]= mensagens

        mensagens.append({
            "usuario": "user",
            "mensagem": mensagem_usuario
        })

        mensagem_ia = perguntando(mensagem_usuario)
        mensagens.append({
            "usuario": "assistant",
            "mensagem": mensagem_ia
        })

        for perfil_e_mensagem in mensagens:

            if perfil_e_mensagem["usuario"] == 'assistant':
                    with st.chat_message(perfil_e_mensagem["usuario"], avatar=company_logo):
                        st.markdown(perfil_e_mensagem["mensagem"])
            
            else:
                with st.chat_message(perfil_e_mensagem["usuario"]):
                    st.markdown(perfil_e_mensagem["mensagem"])

app()