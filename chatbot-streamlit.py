import streamlit as st
import openai
from datetime import datetime
import os
from dotenv import load_dotenv
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pickle

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Intelligo Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KnowledgeBase:
    def __init__(self, api_key=st.secrets["OPENAI_API_KEY"]):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore = None
        self.auto_load_knowledge()
        
    def auto_load_knowledge(self):
        """Automatically load knowledge from knowledge.json file"""
        try:
            # Try to load from knowledge.json in the same directory
            knowledge_file = "knowledge.json"
            if os.path.exists(knowledge_file):
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
                
                # Convert JSON to documents
                documents = self.json_to_documents(knowledge_data)
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                split_docs = text_splitter.split_documents(documents)
                
                # Create FAISS vectorstore
                self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
                print("Knowledge base loaded successfully!")
                return True
            else:
                print("knowledge.json file not found in current directory")
                return False
        except Exception as e:
            print(f"Error auto-loading knowledge base: {str(e)}")
            return False
    
    def load_knowledge_from_json(self, json_file_path):
        """Load knowledge from JSON file and create FAISS vectorstore"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            # Convert JSON to documents
            documents = self.json_to_documents(knowledge_data)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Create FAISS vectorstore
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            return True
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")
            return False
    
    def json_to_documents(self, data, prefix=""):
        """Convert JSON data to LangChain documents"""
        documents = []
        
        def process_item(key, value, current_prefix=""):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    new_prefix = f"{current_prefix}.{key}" if current_prefix else key
                    process_item(sub_key, sub_value, new_prefix)
            elif isinstance(value, list):
                content = f"{current_prefix}.{key}: " + ", ".join(str(item) for item in value)
                documents.append(Document(
                    page_content=content,
                    metadata={"category": current_prefix, "key": key}
                ))
            else:
                content = f"{current_prefix}.{key}: {value}"
                documents.append(Document(
                    page_content=content,
                    metadata={"category": current_prefix, "key": key}
                ))
        
        for key, value in data.items():
            process_item(key, value)
        
        return documents
    
    def search_knowledge(self, query, k=3):
        """Search knowledge base for relevant information"""
        if self.vectorstore is None:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            st.error(f"Error searching knowledge base: {str(e)}")
            return []

class StudentVerification:
    def __init__(self):
        self.required_fields = ["nama", "bootcamp", "batch"]
        
    def is_verified(self, session_state):
        """Check if student is verified"""
        verification_data = session_state.get("student_verification", {})
        return all(verification_data.get(field) for field in self.required_fields)
    
    def get_missing_fields(self, session_state):
        """Get list of missing verification fields"""
        verification_data = session_state.get("student_verification", {})
        return [field for field in self.required_fields if not verification_data.get(field)]
    
    def check_not_registered(self, user_input):
        """Check if user indicates they are not registered"""
        user_input_lower = user_input.lower()
        not_registered_keywords = [
            "belum daftar", "tidak daftar", "belum mendaftar", "tidak mendaftar",
            "belum ikut", "tidak ikut", "belum join", "tidak join",
            "belum terdaftar", "tidak terdaftar", "mau daftar", "ingin daftar",
            "cara daftar", "bagaimana daftar", "pendaftaran"
        ]
        return any(keyword in user_input_lower for keyword in not_registered_keywords)
    
    def extract_verification_info(self, user_input):
        """Extract verification information from user input"""
        info = {}
        user_input_lower = user_input.lower()
        
        # Simple keyword extraction - you can make this more sophisticated
        if "nama" in user_input_lower or "saya" in user_input_lower:
            # Extract name after "nama" or "saya"
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ["nama", "saya"] and i + 1 < len(words):
                    info["nama"] = " ".join(words[i+1:]).strip(".,!?")
                    break
        
        if "bootcamp" in user_input_lower:
            if "data science" in user_input_lower:
                info["bootcamp"] = "Data Science"
            elif "ai" in user_input_lower or "artificial intelligence" in user_input_lower:
                info["bootcamp"] = "AI"
        
        if "batch" in user_input_lower:
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() == "batch" and i + 1 < len(words):
                    info["batch"] = words[i+1].strip(".,!?")
                    break
        
        return info

class ChatAgent:
    def __init__(self, api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = """Kamu adalah AI Assistant chatbot Data Science & AI di Intelligo ID. Kamu membantu siswa dengan pertanyaan teknis dan administrasi terkait bootcamp. Gunakan bahasa Indonesia yang friendly dan profesional."""
        self.knowledge_base = KnowledgeBase(api_key)
        self.student_verification = StudentVerification()
    
    def get_response(self, messages, personality, session_state):
        try:
            if personality == "Mintell Bot":
                return self.get_mintell_response(messages, session_state)
            elif personality == "Student Mentor":
                return self.get_mentor_response(messages, session_state)
            else:
                return self.get_default_response(messages)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_mintell_response(self, messages, session_state):
        """Get response from Mintell Bot with RAG"""
        user_query = messages[-1]["content"]
        
        # Search knowledge base
        relevant_info = self.knowledge_base.search_knowledge(user_query)
        
        # Create enhanced prompt with knowledge
        knowledge_context = "\n".join(relevant_info) if relevant_info else ""
        
        mintell_enhanced_system_prompt = f"""Kamu adalah Mintell Bot, AI Assistant chatbot administrasi di Intelligo ID. 

        KONTEKS PENGETAHUAN:
        {knowledge_context}

        TUGAS UTAMA:
        - Hanya menjawab seputar administrasi bootcamp di Intelligo ID
        - Gunakan informasi dari konteks pengetahuan di atas
        - Jika user bertanya tentang materi Data Science dan AI, arahkan menggunakan model Student Mentor
        - Jangan menjawab selain yang berkaitan dengan Intelligo ID
        - Gunakan bahasa Indonesia yang friendly dan profesional

        Jika informasi tidak tersedia dalam konteks, katakan dengan sopan bahwa Anda perlu menghubungi admin untuk informasi lebih detail."""
        
        # Update system message
        enhanced_messages = messages.copy()
        enhanced_messages[0] = {"role": "system", "content": mintell_enhanced_system_prompt}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=enhanced_messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    def get_mentor_response(self, messages, session_state):
        """Get response from Student Mentor with verification"""
        user_query = messages[-1]["content"]
        
        # Check if student is verified
        if not self.student_verification.is_verified(session_state):
            return self.handle_verification(user_query, session_state)
        
        # Student is verified, proceed with mentoring
        mentor_enhanced_system_prompt = f"""Kamu adalah Student Mentor AI di Intelligo ID. 

        INFORMASI SISWA:
        - Nama: {session_state.get('student_verification', {}).get('nama', 'Tidak diketahui')}
        - Bootcamp: {session_state.get('student_verification', {}).get('bootcamp', 'Tidak diketahui')}
        - Batch: {session_state.get('student_verification', {}).get('batch', 'Tidak diketahui')}

        TUGAS UTAMA:
        - Membantu siswa untuk mentoring seputar Data Science dan AI
        - Jika user menanyakan administrasi bootcamp, arahkan menggunakan model Mintell Bot
        - Jangan menjawab selain Data Science dan AI
        - Gunakan bahasa Indonesia yang friendly dan profesional
        - Panggil siswa dengan nama mereka"""
        
        # Update system message
        enhanced_messages = messages.copy()
        enhanced_messages[0] = {"role": "system", "content": mentor_enhanced_system_prompt}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=enhanced_messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    def handle_verification(self, user_input, session_state):
        """Handle student verification process"""
        # Check if user indicates they are not registered
        if self.student_verification.check_not_registered(user_input):
            return """Terima kasih telah menghubungi Student Mentor Intelligo ID! 

            Sepertinya Anda belum terdaftar di bootcamp kami. Untuk mendaftar bootcamp Data Science atau AI di Intelligo ID, silakan hubungi admin kami:

            📱 **Kontak Admin:**
            • Website: https://www.intelligo.id
            • WhatsApp: Silakan kunjungi website untuk informasi kontak terbaru
            • Atau klik tombol "Hubungi Admin" di website kami

            Admin akan membantu Anda dengan:
            ✅ Informasi lengkap program bootcamp
            ✅ Jadwal kelas dan biaya
            ✅ Proses pendaftaran
            ✅ Opsi beasiswa dan cicilan

            Setelah Anda terdaftar, silakan kembali ke sini untuk mendapatkan mentoring pembelajaran! 🎓"""

        if "student_verification" not in session_state:
            session_state["student_verification"] = {}
        
        # Extract verification info from user input
        extracted_info = self.student_verification.extract_verification_info(user_input)
        
        # Update verification data
        for key, value in extracted_info.items():
            session_state["student_verification"][key] = value
        
        missing_fields = self.student_verification.get_missing_fields(session_state)
        
        if not missing_fields:
            # Verification complete
            nama = session_state["student_verification"]["nama"]
            bootcamp = session_state["student_verification"]["bootcamp"]
            batch = session_state["student_verification"]["batch"]
            return f"""Halo {nama}! ✅ Verifikasi berhasil.

        **Data Anda:**
        • Nama: {nama}
        • Bootcamp: {bootcamp}
        • Batch: {batch}

        Saya siap membantu Anda dengan pembelajaran Data Science dan AI. Apa yang ingin Anda pelajari hari ini? 🎓

        Jika ada pertanyaan administrasi, silakan gunakan Mintell Bot ya!"""
        
        # Still need more information
        current_info = session_state.get("student_verification", {})
        response = """Halo! Saya Student Mentor di Intelligo ID. Sebelum kita mulai mentoring, saya perlu verifikasi data siswa terlebih dahulu.

        **Status Verifikasi:**"""
        
        if current_info.get("nama"):
            response += f"\n✅ Nama: {current_info['nama']}"
        else:
            response += "\n❓ Nama: (belum diisi)"
            
        if current_info.get("bootcamp"):
            response += f"\n✅ Bootcamp: {current_info['bootcamp']}"
        else:
            response += "\n❓ Bootcamp: (belum diisi)"
            
        if current_info.get("batch"):
            response += f"\n✅ Batch: {current_info['batch']}"
        else:
            response += "\n❓ Batch: (belum diisi)"
        
        response += """\n\nSilakan lengkapi informasi yang belum diisi. 
        **Contoh:** "Nama Tatang Suratang, Bootcamp Data Science, batch 15"

        **Atau jika belum terdaftar**, katakan "belum daftar" dan saya akan mengarahkan Anda ke admin untuk pendaftaran."""
        
        return response
    
    def get_default_response(self, messages):
        """Get default response"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    def create_message_history(self, conversations):
        messages = [{"role": "system", "content": self.system_prompt}]
        for conv in conversations:
            messages.append({"role": "user", "content": conv["user"]})
            messages.append({"role": "assistant", "content": conv["assistant"]})
        return messages

def initialize_session_state():
    """Initialize session state variables"""
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False
    if "current_personality" not in st.session_state:
        st.session_state.current_personality = None
    if "student_verification" not in st.session_state:
        st.session_state.student_verification = {}

def sidebar_config():
    """Configure sidebar with settings"""
    st.sidebar.title("⚙️ Configuration")

    # Agent personality
    st.sidebar.subheader("🤖 Assistant Type")
    personality = st.sidebar.selectbox(
        "Choose Assistant",
        ["Mintell Bot", "Student Mentor"],
        help="Choose the assistant type"
    )
    
    # Show verification status for Student Mentor
    if personality == "Student Mentor":
        st.sidebar.subheader("👤 Student Verification")
        verification_data = st.session_state.get("student_verification", {})
        
        if verification_data.get("nama"):
            st.sidebar.success(f"✅ Verified: {verification_data['nama']}")
            st.sidebar.info(f"Bootcamp: {verification_data.get('bootcamp', 'N/A')}")
            st.sidebar.info(f"Batch: {verification_data.get('batch', 'N/A')}")
        else:
            st.sidebar.warning("⚠️ Not verified yet")
        
        if st.sidebar.button("Reset Verification"):
            st.session_state.student_verification = {}
            st.rerun()
    
    # Show knowledge base status for Mintell Bot
    if personality == "Mintell Bot":
        st.sidebar.subheader("📚 Knowledge Base")
        if st.session_state.agent and st.session_state.agent.knowledge_base.vectorstore:
            st.sidebar.success("✅ Knowledge base loaded")
        else:
            st.sidebar.info("📚 Knowledge base will auto-load")
    
    if personality:
        if st.sidebar.button("Initialize Agent") or st.session_state.current_personality != personality:
            try:
                if st.session_state.agent is None:
                    st.session_state.agent = ChatAgent()
                
                st.session_state.api_key_set = True
                st.session_state.current_personality = personality
                st.sidebar.success(f"{personality} initialized successfully!")
                
                # Clear conversations when switching personality
                if len(st.session_state.conversations) > 0:
                    st.session_state.conversations = []
                    
            except Exception as e:
                st.sidebar.error(f"Error initializing agent: {str(e)}")
    
    # Chat controls
    st.sidebar.markdown("---")
    st.sidebar.title("💬 Chat Controls")
    
    if st.sidebar.button("Clear Chat History"):
        st.session_state.conversations = []
        if st.session_state.current_personality == "Student Mentor":
            st.session_state.student_verification = {}
        st.rerun()
    
    if st.sidebar.button("Export Chat"):
        if st.session_state.conversations:
            chat_data = {
                "timestamp": datetime.now().isoformat(),
                "personality": st.session_state.current_personality,
                "conversations": st.session_state.conversations,
                "verification_data": st.session_state.get("student_verification", {})
            }
            st.sidebar.download_button(
                "Download Chat History",
                data=json.dumps(chat_data, indent=2, ensure_ascii=False),
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def main_chat_interface():
    """Main chat interface"""
    current_personality = st.session_state.get("current_personality", "None")
    st.title(f"🎓 AI Chatbot Intelligo Assistant - {current_personality}")
    
    if current_personality == "Mintell Bot":
        if st.session_state.agent and st.session_state.agent.knowledge_base.vectorstore:
            st.success("📚 Knowledge base loaded - Ready to answer administrative questions!")
        else:
            st.info("📚 Knowledge base will be automatically loaded when initialized")
    elif current_personality == "Student Mentor":
        verification_status = "✅ Verified" if st.session_state.get("student_verification", {}).get("nama") else "❌ Not Verified"
        st.info(f"👤 Student Status: {verification_status}")
    
    st.markdown("Welcome to your AI chatbot assistant! Start a conversation below.")
    
    # Check if agent is initialized
    if not st.session_state.api_key_set or st.session_state.agent is None:
        st.warning("Please initialize your agent in the sidebar to start chatting.")
        return
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, conv in enumerate(st.session_state.conversations):
            # User message
            with st.chat_message("user"):
                st.write(conv["user"])
                st.caption(f"🕒 {conv['timestamp']}")
            
            # Assistant message
            with st.chat_message("assistant"):
                st.write(conv["assistant"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_input)
            st.caption(f"🕒 {timestamp}")
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare message history
                message_history = st.session_state.agent.create_message_history(
                    st.session_state.conversations
                )
                message_history.append({"role": "user", "content": user_input})
                
                # Get response with personality and session state
                response = st.session_state.agent.get_response(
                    message_history, 
                    st.session_state.current_personality,
                    st.session_state
                )
                
                # Display response
                st.write(response)
        
        # Add to conversation history
        st.session_state.conversations.append({
            "user": user_input,
            "assistant": response,
            "timestamp": timestamp
        })
        
        # Rerun to update the interface
        st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    sidebar_config()
    main_chat_interface()

if __name__ == "__main__":
    main()