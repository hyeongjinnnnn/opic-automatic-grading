from langchain_community.llms import HuggingFaceHub
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_models.huggingface import ChatHuggingFace
import json

rubric = {
    "NO": "Novice-level speakers can communicate short messages on highly predictable, everyday topics that affect them directly. They do so primarily through the use of isolated words and phrases that have been encountered, memorized, and recalled. Novice-level speakers may be difficult to understand even by the most sympathetic interlocutors accustomed to non-native speech",
    "IL": "Speakers at the Intermediate Low sublevel are able to handle successfully a limited number of uncomplicated communicative tasks by creating with the language in straightforward social situations. Conversation is restricted to some of the concrete exchanges and predictable topics necessary for survival in the target-language culture. These topics relate to basic personal information; for example, self and family, some daily activities and personal preferences, and some immediate needs, such as ordering food and making simple purchases. At the Intermediate Low sublevel, speakers are primarily reactive and struggle to answer direct questions or requests for information. They are also able to ask a few appropriate questions. Intermediate Low speakers manage to sustain the functions of the Intermediate level, although just barely.",
    "IM": "Mid Speakers at the Intermediate Mid sublevel are able to handle successfully a variety of uncomplicated communicative tasks in straightforward social situations. Conversation is generally limited to those predictable and concrete exchanges necessary for survival in the target culture. These include personal information related to self, family, home, daily activities, interests and personal preferences, as well as physical and social needs, such as food, shopping, travel, and lodging. Intermediate Mid speakers tend to function reactively, for example, by responding to direct questions or requests for information. However, they are capable of asking a variety of questions when necessary to obtain simple information to satisfy basic needs, such as directions, prices, and services. When called on to perform functions or handle topics at the Advanced level, they provide some information but have difficulty linking ideas, manipulating time and aspect, and using communicative strategies, such as circumlocution. Intermediate Mid speakers are able to express personal meaning by creating with the language, in part by combining and recombining known elements and conversational input to produce responses typically consisting of sentences and strings of sentences. Their speech may contain pauses, reformulations, and self-corrections as they search for adequate vocabulary and appropriate language forms to express themselves. In spite of the limitations in their vocabulary and/or pronunciation and/or grammar and/or syntax, Intermediate Mid speakers are generally understood by sympathetic interlocutors accustomed to dealing with non-natives.",
    "IH": "Intermediate High speakers are able to converse with ease and confidence when dealing with the routine tasks and social situations of the Intermediate level. They are able to handle successfully uncomplicated tasks and social situations requiring an exchange of basic information related to their work, school, recreation, particular interests, and areas of competence. Intermediate High speakers can handle a substantial number of tasks associated with the Advanced level, but they are unable to sustain performance of all of these tasks all of the time. Intermediate High speakers can narrate and describe in all major time frames using connected discourse of paragraph length, but not all the time. Typically, when Intermediate High speakers attempt to perform Advanced-level tasks, their speech exhibits one or more features of breakdown, such as the failure to carry out fully the narration or description in the appropriate major time frame, an inability to maintain paragraph-length discourse, or a reduction in breadth and appropriateness of vocabulary. Intermediate High speakers can generally be understood by native speakers unaccustomed to dealing with non-natives, although interference from another language may be evident (e.g., use of code-switching, false cognates, literal translations), and a pattern of gaps in communication may occur.",
    "AL": "Speakers at the Advanced Low sublevel are able to handle a variety of communicative tasks. They are able to participate in most informal and some formal conversations on topics related to school, home, and leisure activities. They can also speak about some topics related to employment, current events, and matters of public and community interest. Advanced Low speakers demonstrate the ability to narrate and describe in the major time frames of past, present, and future in paragraph-length discourse with some control of aspect. In these narrations and descriptions, Advanced Low speakers combine and link sentences into connected discourse of paragraph length, although these narrations and descriptions tend to be handled separately rather than interwoven. They can handle appropriately the essential linguistic challenges presented by a complication or an unexpected turn of events. Responses produced by Advanced Low speakers are typically not longer than a single paragraph. The speaker’s dominant language may be evident in the use of false cognates, literal translations, or the oral paragraph structure of that language. At times their discourse may be minimal for the level, marked by an irregular flow, and containing noticeable self-correction. More generally, the performance of Advanced Low speakers tends to be uneven. Advanced Low speech is typically marked by a certain grammatical roughness (e.g., inconsistent control of verb endings), but the overall performance of the Advanced-level tasks is sustained, albeit minimally. The vocabulary of Advanced Low speakers often lacks specificity. Nevertheless, Advanced Low speakers are able to use communicative strategies such as rephrasing and circumlocution. Advanced Low speakers contribute to the conversation with sufficient accuracy, clarity, and precision to convey their intended message without misrepresentation or confusion. Their speech can be understood by native speakers unaccustomed to dealing with non-natives, even though this may require some repetition or restatement. When attempting to perform functions or handle topics associated with the Superior level, the linguistic quality and quantity of their speech will deteriorate significantly."
}

question = "Can you introduce yourself in detail as much as possible?"
stt = "Hello there! I'm an AI conversational chatbot, designed to be an advanced language model. I've been trained on a massive dataset of text and code, allowing me to comprehend and generate human-like text in response to your queries."
ratings = {'Task_Completion': 4.50, 'Accuracy': 5.00, 'Appropriateness': 5.00}
final_level = 'AL'

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
        "return_full_text": False,
    },
    huggingfacehub_api_token="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"
)

messages = [
    SystemMessage(content=
                  f"""
                  You're a professional English teacher.

                  Your task is to provide feedback based on the questions and answers, determining whether the answer is an appropriate answer to the question.
                  keep in mind that a few spelling errors can be included in native responses with a small portion.\n\n
                  Note that the given response is a transcribed English text and given as a pair with a question of item and please keep in mind that this is a English oral test, which means that you should evaluate the response as an oral test.
                  
                  [Input Data]
                  Question: {question}
                  STT: {stt}
                  """
                  ),
    HumanMessage(
        content="""
        [Request]
                  1. Please briefly write the overall feedback (strengths, weaknesses, directions for improvement, etc.) based on the ratings(1~5) and evaluation rubric.
                  2. Please provide feedback in Korean
        """
    ),
]

chat_model = ChatHuggingFace(llm=llm)
res = chat_model.invoke(messages)
print(res.content)

