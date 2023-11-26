import openai


class OpenaiCompletionBase:
    def __init__(self, model_name="text-davinci-003",
                 api_version="2023-05-15", temperature=0.0, max_tokens=200, prompt_template=""):
        self.model_name = model_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template
        self.top_p = 1
        self.n = 1

    def _get_response(self, **input_args):
        prompt = self.prompt_template.format(**input_args)
        response = openai.Completion.create(
            engine=self.model_name,
            api_version=self.api_version,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=self.n
        )

        return response

    def call_llm(self, input_text):
        input_args = {"input_text": input_text}
        response = self._get_response(**input_args)
        output_text = response["choices"][0]["text"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        input_args = {"input_text": input_text}
        response = self._get_response(**input_args)
        output_text = response["choices"][0]["text"].strip()
        return output_text, response

    def get_config(self):
        return self.__dict__


class OpenaiCompletionWordLimit(OpenaiCompletionBase):
    def __init__(self, model_name="text-davinci-003",
                 api_version="2023-05-15", temperature=0.0, max_tokens=200, prompt_template="", max_word_ratio=0.6):
        super().__init__(model_name, api_version, temperature, max_tokens, prompt_template)
        self.max_word_ratio = max_word_ratio

    def call_llm(self, input_text):
        input_text_word_count = len(input_text.split())
        max_words = int(round(input_text_word_count * self.max_word_ratio))
        input_args = {"max_words": max_words, "input_text": input_text}
        response = self._get_response(**input_args)
        output_text = response["choices"][0]["text"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        input_text_word_count = len(input_text.split())
        max_words = int(round(input_text_word_count * self.max_word_ratio))
        input_args = {"max_words": max_words, "input_text": input_text}
        response = self._get_response(**input_args)
        output_text = response["choices"][0]["text"].strip()
        return output_text, response


class OpenaiChatCompletionBase:
    def __init__(self, model_name="gpt-3-5-turbo", api_version="2023-05-15", temperature=0.0, max_tokens=200,
                 system_prompt_template="", user_prompt_template=""):
        self.model_name = model_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.top_p = 1
        self.n = 1

    def _get_response(self, messages):
        response = openai.ChatCompletion.create(
            engine=self.model_name,
            api_version=self.api_version,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=self.n
        )

        return response

    def get_config(self):
        return self.__dict__


class OpenaiChatCompletionWordLimit(OpenaiChatCompletionBase):
    def __init__(self, model_name="gpt-3-5-turbo", api_version="2023-05-15", temperature=0.0, max_tokens=200,
                 system_prompt_template="", user_prompt_template="", max_word_ratio=0.6):
        super().__init__(model_name, api_version, temperature, max_tokens, system_prompt_template, user_prompt_template)
        self.max_word_ratio = max_word_ratio

    def call_llm(self, input_text):
        input_text_word_count = len(input_text.split())
        max_words = int(round(input_text_word_count * self.max_word_ratio))
        system_prompt = self.system_prompt_template.format(**{'max_words': max_words})
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        input_text_word_count = len(input_text.split())
        max_words = int(round(input_text_word_count * self.max_word_ratio))
        system_prompt = self.system_prompt_template.format(**{'max_words': max_words})
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text, response


class OpenaiChatCompletionNRephraser(OpenaiChatCompletionBase):
    def __init__(self, model_name="gpt-3-5-turbo", api_version="2023-05-15", temperature=0.0, max_tokens=200,
                 system_prompt_template="", user_prompt_template="", localization_prompt_template="", style='casual',
                 n_paraphrase=1):
        super().__init__(model_name, api_version, temperature, max_tokens, system_prompt_template, user_prompt_template)
        self.localization_prompt_template = localization_prompt_template
        self.style = style
        self.n_paraphrase = n_paraphrase

    def set_system_prompt(self, text):
        localization_prompt = self.localization_prompt_template.format(**{'input_text': text[:30]})
        system_args = {
            'language_input': localization_prompt,
            'style': self.style,
            'n_paraphrase': self.n_paraphrase
        }
        return self.system_prompt_template.format(**system_args)

    def call_llm(self, input_text):
        system_prompt = self.set_system_prompt(input_text)
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        system_prompt = self.set_system_prompt(input_text)
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text, response

class OpenaiChatCompletionRephraser(OpenaiChatCompletionBase):
    def __init__(self, model_name="gpt-3-5-turbo", api_version="2023-05-15", temperature=0.0, max_tokens=200,
                 system_prompt_template="", user_prompt_template="", localization_prompt_template="", style='casual'):
        super().__init__(model_name, api_version, temperature, max_tokens, system_prompt_template, user_prompt_template)
        self.localization_prompt_template = localization_prompt_template
        self.style = style

    def set_system_prompt(self, text):
        localization_prompt = self.localization_prompt_template.format(**{'input_text': text[:30]})
        system_args = {
            'language_input': localization_prompt,
            'style': self.style,
        }
        return self.system_prompt_template.format(**system_args)

    def call_llm(self, input_text):
        system_prompt = self.set_system_prompt(input_text)
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        system_prompt = self.set_system_prompt(input_text)
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text, response

class OpenaiChatCompletionRephraserNoTone(OpenaiChatCompletionBase):
    def __init__(self, model_name="gpt-3-5-turbo", api_version="2023-05-15", temperature=0.0, max_tokens=200,
                 system_prompt_template="", user_prompt_template="", localization_prompt_template=""):
        super().__init__(model_name, api_version, temperature, max_tokens, system_prompt_template, user_prompt_template)
        self.localization_prompt_template = localization_prompt_template

    def set_system_prompt(self, text):
        if self.localization_prompt_template:
            localization_prompt = self.localization_prompt_template.format(**{'input_text': text[:30]})
            system_args = {
                'language_input': localization_prompt,
            }
            return self.system_prompt_template.format(**system_args)
        return self.system_prompt_template

    def call_llm(self, input_text):
        system_prompt = self.set_system_prompt(input_text)
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        system_prompt = self.set_system_prompt(input_text)
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text, response


class OpenaiChatCompletionCurrentRephraser(OpenaiChatCompletionBase):
    def __init__(self, model_name="gpt-3-5-turbo", api_version="2023-05-15", temperature=0.0, max_tokens=200,
                 system_prompt_template="", user_prompt_template="", localization_prompt_template="", style='casual',
                 n_paraphrase=1, n=1):
        super().__init__(model_name, api_version, temperature, max_tokens, system_prompt_template, user_prompt_template)
        self.localization_prompt_template = localization_prompt_template
        self.style = style
        self.n_paraphrase = n_paraphrase
        self.n = n

    def set_user_prompt(self, text):
        localization_prompt = self.localization_prompt_template.format(**{'input_text': text[:30]})
        prompt_args = {
            'language_input': localization_prompt,
            'style': self.style,
            'input_text': text
        }
        return self.user_prompt_template.format(**prompt_args)

    def call_llm(self, input_text):
        user_prompt = self.set_user_prompt(input_text)
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        user_prompt = self.set_user_prompt(input_text)
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text, response
