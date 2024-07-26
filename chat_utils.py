import streamlit as st
import autogen

from autogen import OpenAIWrapper, AssistantAgent, UserProxyAgent
from typing import List


def setting_config(model_name: str) -> tuple[List[dict], dict]:
    if model_name.startswith("gpt"):
        config_list = [
            {
                "model": model_name, 
                "base_url": "https://api.openai.com/v1", 
                "api_key": st.secrets.openai.api_key, 
            }
        ]
    else:
        config_list = [
            {
                "model": model_name, # "google/gemma-2-9b-it",
                "base_url": "https://integrate.api.nvidia.com/v1", 
                "api_key": st.secrets.nvidia.api_key, 
                "max_tokens": 1000, 
            }
        ]

    llm_config={
        "timeout": 500, 
        "seed": 42, 
        "config_list": config_list, 
        "temperature": 0.5, 
    }

    return config_list, llm_config


class TrackableUserProxyAgent(UserProxyAgent):
    t = 0
    def _process_received_message(self, message, sender, silent):
        global t
        with st.chat_message(sender.name, avatar="streamlit_images/{}.png".format(self.t)):
            st.write(f"**{message['name'].replace('_',' ')}**: {message['content']}")
            self.t += 1
            if self.t == 4:
                self.t = 0
        st.divider()
        
        return super()._process_received_message(message, sender, silent)


def doctor_recruiter(prompt_recruiter_doctors: str, model_name: str) -> str:
    config_list, llm_config = setting_config(model_name) 
    client = OpenAIWrapper(api_key=config_list[0]['api_key'], config_list=config_list)
    response = client.create(messages=[{"role": "user", "content": prompt_recruiter_doctors + "\nReturn ONLY JSON file (don't use Markdown tags like: ```json)."}], 
                            temperature=0.3, 
                            seed=42, 
                            model=config_list[0]['model'])
    text = client.extract_text_or_completion_object(response)

    return text


def doctor_discussion(doctor_description: str, prompt_internist_doctor: str, model_name: str) -> str:
    config_list, llm_config = setting_config(model_name)
    doctor = OpenAIWrapper(api_key=config_list[0]['api_key'], config_list=config_list)
    response = doctor.create(messages=[
                                        {"role": "assistant", 
                                        "content": doctor_description},
                                        {"role": "user", 
                                        "content": prompt_internist_doctor + "\nDon't use Markdown tags."}
                                        ], 
                            temperature=0.3, 
                            model=config_list[0]['model'])
    text = doctor.extract_text_or_completion_object(response)

    return text


def multiagent_doctors(json_data: dict, model_name: str) -> List[AssistantAgent]:
    config_list, llm_config = setting_config(model_name)
    doc = []
    for i in range(len(json_data['doctors'])):
        doc.append(AssistantAgent(
            name=json_data['doctors'][i]['role'].replace(" ", "_"), 
            llm_config=llm_config, 
            system_message="As a " + json_data['doctors'][i]['role'].replace(" ", "_") + ". Discuss with other medical experts in the team to help the INTERNIST DOCTOR make a final decision. Avoid postponing further examinations and repeating opinions given in the analysis, but explain in a logical and concise manner why you are making this final decision."))

    return doc


def care_discussion_start(doc: List[AssistantAgent], prompt_reunion: str, internist_sys_message: str, model_name: str) -> autogen.GroupChatManager:
    config_list, llm_config = setting_config(model_name)
    doc.append(TrackableUserProxyAgent(
        name="internist_doctor", 
        human_input_mode="NEVER", 
        max_consecutive_auto_reply=1, 
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(("JUSTIFIABLE", "UNJUSTIFIABLE")), 
        code_execution_config=False, 
        llm_config=llm_config, 
        system_message=internist_sys_message 
    ))

    groupchat = autogen.GroupChat(agents=doc, 
                                messages=[], 
                                max_round=(len(doc)+1), 
                                speaker_selection_method="round_robin", 
                                role_for_select_speaker_messages='user', 
                                )

    manager = autogen.GroupChatManager(groupchat=groupchat, 
                                    llm_config=llm_config, 
                                    max_consecutive_auto_reply=1 
                                    )

    doc[-1].initiate_chat(
        manager,
        message=prompt_reunion,
    )

    return manager