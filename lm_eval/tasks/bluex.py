"""
University Entrance Exam as a Guiding Test for Artificial Intelligence
https://www.ime.usp.br/~ddm/project/enem/ENEM-GuidingTest.pdf

The ENEM Challenge consists in designing an autonomous system that matches the 
performance of a human students on the exam. The overall goal is to foster and 
evaluate the development of Artificial Intelligence techniques that have good 
performance on complex cognitive tasks, not particularly designed for AI systems. 
In addition, this challenge aims to promote and give more visiblity to the 
development of NLP tools for Brazilian Portuguese.

Homepage: https://www.ime.usp.br/~ddm/project/enem
"""
import collections
from io import BytesIO
import json
import numpy as np
import os
import re
from urllib.request import urlopen
import xml.etree.ElementTree as ET 
from zipfile import ZipFile
from datasets import load_dataset
from lm_eval import utils
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from fastchat.conversation import get_conv_template
import re


_CITATION = """
@InProceedings{ ENEM-Challenge,
    author={Silveira, Igor Cataneo and Mau\'a, Denis Deratani},
    booktitle={Proceedings of the 6th Brazilian Conference on Intelligent Systems},
    series={BRACIS},
    title={University Entrance Exam as a Guiding Test for Artificial Intelligence},
    pages={426--431},
    year={2017}
}
"""


PATTERNS_REPLACES = [
    (r'\s*\n+\s*', r' '),  # changing \n to space
    (r'(\s)\1+', r' '),  # changing \n to space
    (r'^\s+', r''),
]


apply_regex = lambda pattern, replace, text: re.sub(pattern, replace, text)
# blind 

from enum import Enum
# enumerator with blind, image and caption 
class BLUEX_EVAL_MODE:
    IGNORE_IMAGES = 0
    IMAGE = 1
    BLIND_CAPTION = 2
    CONTEXT_CAPTION = 3



class BLUEX(Task):
    VERSION = 0
    DATASET_PATH = 'portuguese-benchmark-datasets/BLUEX_test'
    DATASET_NAME = "default"
    COT = False
    mode= BLUEX_EVAL_MODE.IGNORE_IMAGES



    use_just_linguistic_and_humanities = False
    tag = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # download and process dataset
        dataset = load_dataset(self.DATASET_PATH,self.DATASET_NAME)["questions"]


        self.dataset = collections.defaultdict(list)

        for example in dataset:
            question_data = example.copy()
            year = example["id"].split("_")[1]

            question_data["test_year"] = year

            if example["id"] in ["UNICAMP_2024_40"]:
                continue

            # skip questions with null answers ( i.e the question was cancelled in the original test)
            if question_data["answer"] == None:
                continue

            # skip questions that have images as answers
            if question_data["alternatives_type"]=="images":
                continue

            # substitute multiple spaces to single space

            if "context_captions" not in question_data:
                print(question_data["id"])
                raise

            for pattern, replace in PATTERNS_REPLACES:
                question_data["question"] = re.sub(
                    pattern, replace, question_data["question"]
                )
                for question_id in range(len(question_data["alternatives"])):
                    question_data["alternatives"][question_id] = re.sub(
                        pattern, replace, question_data["alternatives"][question_id]
                    )

            

            self.dataset["test"].append(question_data)

        self.dataset["test"] = list(map(self._process_doc, self.dataset["test"]))

    def _get_train_examples(self):
        header = 'Urgência emocional. Se tudo é para ontem, se a vida engata uma primeira e sai em disparada, se não há mais tempo para paradas estratégicas, caímos fatalmente no vício de querer que os amores sejam igualmente resolvidos num átimo de segundo. Temos pressa para ouvir "eu te amo". Não vemos a hora de que fiquem estabelecidas as regras de convívio: somos namorados, ficantes, casados, amantes? Urgência emocional. Uma cilada. Associamos diversas palavras ao AMOR: paixão, romance, sexo, adrenalina, palpitação. Esquecemos, no entanto, da palavra que viabiliza esse sentimento: "paciência". Amor sem paciência não vinga. Amor não pode ser mastigado e engolido com emergência, com fome desesperada. É uma refeição que pode durar uma vida. MEDEIROS, M. Disponível em: http://porumavidasimples.blogspot.com.br. Acesso em: 20 ago. 2017 (adaptado).'
        statement = 'Nesse texto de opinião, as marcas linguísticas revelam uma situação distensa e de pouca formalidade, o que se evidencia pelo(a) '
        options = [
            'A. impessoalização ao longo do texto, como em: "se não há mais tempo". ',
            'B. construção de uma atmosfera de urgência, em palavras como: "pressa". ',
            'C. repetição de uma determinada estrutura sintática, como em: "Se tudo é para ontem". ',
            'D. ênfase no emprego da hipérbole, como em: "uma refeição que pode durar uma vida". ',
            'E. emprego de metáforas, como em: "a vida engata uma primeira e sai em disparada". ',
        ]
        explanation_1 = 'A alternativa A. está ERRADA porque impessoalização não é uma marca de pouca formalidade, inclusive o uso do verbo haver representa uma marca de formalidade. A alternativa B. está ERRADA porque o texto até criou uma atmosfera de urgência, embora tenha sido para criticá-la, e discute exatamente a importância da paciência e não da pressa. A alternativa C. está ERRADA porque a estrutura sintática não é repetida sistematicamente ao longo do texto. A alternativa D. está ERRADA porque, embora o texto possua hipérboles, para afirmar que a figura de linguagem é enfatizada, ela deveria aparecer mais vezes. A alternativa E. está CORRETA porque o texto possui comparações implícitas que se caracterizam como metáforas. Logo o texto emprega metáforas. Resposta:'
        explanation_2 = 'O texto é escrito em uma linguagem leve, ágil, e de pouca formalidade. Além disso, possui figuras de linguagem, como metáforas e hipérboles, que não são excludentes. Em uma análise sequencial das alternativas, daria para afirmar que D. e E. estão corretas. Entretanto, observando em detalhes, nota-se que a expressão "emprego de metáforas" mostra ser mais adequada do que "ênfase no emprego da hipérbole", visto que, para afirmarmos que o uso de hipérboles foi enfatizado, a figura de linguagem deveria ter aparecido mais vezes. Isso torna a alternativa E. mais provável de ser CORRETA. Além disso, impessoalização não deve ser apontada como marca de pouca formalidade. Existe também uma atmosfera de urgência, mas que é criticada no texto que destaca a importância da paciência e não da pressa. Por fim, a estrutura sintática não é repetida sistematicamente ao longo do texto. Resposta:'
        document_1 = {
            'id': 'ENEM_2022_21',  # used to filter out from test set
            'exam': '2022',  # used to get metrics for each exam, and to filter out prompt candidates
            'question': header+"\n"+statement,
            'alternatives': options,
            'answer': 'e',
            'explanation': explanation_2,
            "subject": [],
            "test_year": 2022,
            "IU": False,
            "BK": False,
            "MR": False,
            "associated_images": [],
            "blind_captions": [],
            "context_captions": []
        }

        header = 'Sempre que a relevância do discurso entra em jogo, a questão torna-se política por definição, pois é o discurso que faz do homem um ser político. E tudo que os homens fazem, sabem ou experimentam só tem sentido na medida em que pode ser discutido. Haverá, talvez, verdades que ficam além da linguagem e que podem ser de grande relevância para o homem no singular, isto é, para o homem que, seja o que for, não é um ser político. Mas homens no plural, isto é, os homens que vivem e se movem e agem neste mundo, só podem experimentar o significado das coisas por poderem falar e ser inteligíveis entre si e consigo mesmos. ARENDT, H. A condição humana. Rio de Janeiro: Forense Universitária, 2004.'
        statement = 'No trecho, a filósofa Hannah Arendt mostra a importância da linguagem no processo de'
        options = [
            'A. entendimento da cultura.',
            'B. aumento da criatividade.',
            'C. percepção da individualidade.',
            'D. melhoria da técnica.',
            'E. construção da sociabilidade.',
        ]
        explanation_1 = 'A alternativa A. está ERRADA porque Hannah Arendt não trata do entendimento da cultura, mas da relação social entre as pessoas dessa cultura. A alternativa B. está ERRADA porque Hannah Arendt não fala sobre criatividade, mas sobre a construção de laços entre as pessoas. A alternativa C. está ERRADA porque a linguagem é utilizada no oposto da individualidade, em algo mais coletivo e social. A alternativa D. está ERRADA porque o texto não fala de técnica, mas de laços. A alternativa E. está CORRETA porque a nossa sociabilidade se constrói a partir da linguagem, o que faz de nós seres políticos, no sentido de viver em sociedade, em ambientes coletivos. Resposta:'
        explanation_2 = 'Hannah Arendt defende em sua obra que somos seres políticos, no sentido próprio de vivermos em pólis, em ambiente coletivo e social. E essa sociabilidade só é possível por meio do discurso, da linguagem. Desse modo, podemos concluir que a linguagem se apresenta como uma importante ferramenta para a construção da sociabilidade, e portanto a alternativa E. é a CORRETA. Além disso, não se trata do entendimento da cultura, mas da relação social entre as pessoas dessa cultura. Hannah também não fala sobre aumento de criatividade, tampouco sobre técnica. Por fim, a linguagem é utilizada em algo mais coletivo e social, justamente o oposto da individualidade. Resposta:'
        document_2 = {
            'id': 'ENEM_2022_88',  # used to filter out from test set
            'exam': '2022',  # used to get metrics for each exam, and to filter out prompt candidates
            'question': header+"\n"+statement,
            'alternatives': options,
            'answer': 'e',
            'explanation': explanation_2,
            "subject": [],
            "test_year": 2022,
            "IU": False,
            "BK": False,
            "MR": False,
            "associated_images": [],
            "blind_captions": [],
            "context_captions": []

        }

        header = 'Um casal planeja construir em sua chácara uma piscina com o formato de um paralelepípedo reto retângulo com capacidade para 90 000 L de água. O casal contratou uma empresa de construções que apresentou cinco projetos com diferentes combinações nas dimensões internas de profundidade, largura e comprimento. A piscina a ser construída terá revestimento interno em suas paredes e fundo com uma mesma cerâmica, e o casal irá escolher o projeto que exija a menor área de revestimento. As dimensões internas de profundidade, largura e comprimento, respectivamente, para cada um dos projetos, são: projeto I: 1,8 m, 2,0 m e 25,0 m; projeto II: 2,0 m, 5,0 m e 9,0 m; projeto III: 1,0 m, 6,0 m e 15,0 m; projeto IV: 1,5 m, 15,0 m e 4,0 m; projeto V: 2,5 m, 3,0 m e 12,0 m.'
        statement = 'O projeto que o casal deverá escolher será o'
        options = [
            'A. I.',
            'B. II.',
            'C. III.',
            'D. IV.',
            'E. V.',
        ]
        explanation_1 = 'Devemos calcular a área das quatro faces laterais e a área da base inferior (fundo da piscina) e somar essas áreas para obter a área de revestimento. Logo, calculando a área de revestimento de cada projeto, temos: Projeto I: A = 2 x 25 + 2 x 1,8 x (2 + 25) = 147,2; Projeto II: A = 9 x 5 + 2 x 2 x (9 + 5) = 101; Projeto III: A = 15 x 6 + 2 x 1 x (15 + 6) = 132; Projeto IV: A = 4 x 15 + 2 x 1,5 x (15 + 4) = 117; Projeto V: A = 3 x 12 + 2 x 2,5 x (3 + 12) = 111. Logo, o projeto com menor área de revestimento, é o projeto II, portanto a resposta corrreta é B. Resposta:'
        explanation_2 = 'Devemos calcular a área das quatro faces laterais e a área da base inferior (fundo da piscina) e somar essas áreas para obter a área de revestimento. Logo, calculando a área de revestimento de cada projeto, temos: Projeto I: A = 2 x 25 + 2 x 1,8 x (2 + 25) = 147,2; Projeto II: A = 9 x 5 + 2 x 2 x (9 + 5) = 101; Projeto III: A = 15 x 6 + 2 x 1 x (15 + 6) = 132; Projeto IV: A = 4 x 15 + 2 x 1,5 x (15 + 4) = 117; Projeto V: A = 3 x 12 + 2 x 2,5 x (3 + 12) = 111. Logo, o projeto com menor área de revestimento, é o projeto II, portanto a resposta correta é B. Resposta:'
        document_3 = {
            'id': 'ENEM_2022_143',  # used to filter out from test set
            'exam': '2022',  # used to get metrics for each exam, and to filter out prompt candidates
            'question': header+"\n"+statement,
            'alternatives': options,
            'answer': 'b',
            'explanation': explanation_2,
            "subject": [],
            "test_year": 2022,
            "IU": False,
            "BK": False,
            "MR": False,
            "associated_images": [],
            "blind_captions": [],
            "context_captions": []
        }
        return [document_1, document_2, document_3]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True
        
    def training_docs(self):
        return list(map(self._process_doc, self._get_train_examples()))

    def test_docs(self):
        return self.dataset["test"]

    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
            Enunciado: <enunciado>
            nAlternativas:
            A. <Alternativa1>
            B. <Alternativa2>
            C. <Alternativa3>
            D. <Alternativas4>
            Resposta:
            """
            prompt = "Enunciado: " + doc["question"] + "\nAlternativas:\n"

            alternative_mappings= (
                ('a)', 'A.'),
                ('b)', 'B.'),
                ('c)', 'C.'),
                ('d)', 'D.'),
                ('e)', 'E.'),
            )
            for alternative in doc["alternatives"]:

                alternative_letter=alternative[:5]
                rest=alternative[5:]
                for mapping in alternative_mappings:
                    alternative_letter=alternative_letter.replace(mapping[0], mapping[1])

                alternative_text=alternative_letter+rest
                prompt += f"{alternative_text}\n"

            if self.COT:
                prompt += "Explicação:" + doc.get("explanation", "")
            else:
                prompt += "Resposta:"
            return prompt

        university = doc["id"].split("_")[0]
        if university == "UNICAMP":
            choices = ["a", "b", "c", "d"]
        else:
            choices = ["a", "b", "c", "d", "e"]
        return {
            "query": format_example(doc, choices),
            "choices": doc["alternatives"],
            "gold": choices.index(doc["answer"].lower()),
            "id": doc["id"],
            "exam": doc["test_year"],
            # BK stands for Brazilian Knowledge and is true when the question
            # is strongly related to the Brazilian history, culture, geography, literature, etc.
            "BK": doc["BK"],
            "IU": doc["IU"],
            "MR": doc["MR"],
            "contents": doc["question"],  # used for indexing
            "subject": doc["subject"] if "subject" in doc else [],
            "university": university,
            "associated_images": doc["associated_images"],
            "has_associated_images": len(doc["associated_images"]) > 0,
            "blind_captions": doc["blind_captions"],
            "context_captions": doc["context_captions"],
            "explanation": doc.get("explanation", "")
        }

    def doc_to_text(self, doc):
        return doc["query"]


    
    def doc_to_target(self, doc):
        if self.COT and doc.get("explanation", ""):
            return f"Explicação: {doc['explanation']} " + ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']].upper()
        return " " + ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']].upper()

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        continuation = rf.greedy_until(ctx, ['\n##\n'])
        return continuation

    def process_results(self, doc, results):
        gold = ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']]
        pred = results[0]

        print(doc["query"])

        # regex processing. Useful for zero-shot
        match_1 = re.findall(r'(?:|[Ll]etra |[Aa]lternativa )([ABCDE])\.', pred)
        match_2 = re.findall(r'(?:|[Ll]etra |[Aa]lternativa )([ABCDEabcde])', pred)
        match_3 = re.findall(r'(?:[Rr]esposta: )(\(?[ABCDEabcde][\.\)])', pred)

        if len(match_1) > 0:
            pred = match_1[-1] + '.'
        elif len(match_2) > 0:
            pred = match_2[-1].upper() + '.'
        elif len(match_3) > 0:
            pred = match_3[-1].upper()
            if ")" in pred or "(" in pred:
                pred = pred.replace(")", "")
                pred = pred.replace("(", "")
            if not pred.endswith("."):
                pred = pred + "."
        else:
            print(f'Regex failed at processing {pred=}')
            print(f'{gold=}, {pred=}, {doc["exam"]=}')

        #print(pred, gold)
        acc = 1. if pred == gold else 0.

        results = {
            "acc": acc,
            doc["exam"]: acc,
            "debug_info": {
                "expected_answer": gold,
                "predicted_answer": pred,
            },
        }

        if doc["has_associated_images"]:
            results["has_associated_images"] = acc

        if doc["BK"]:
            results["BK"] = acc

        if doc["IU"]:
            results["IU"] = acc

        if doc["university"]:
            results[doc["university"]] = acc

        for sub in doc["subject"]:
            results[sub] = acc

        return results
    
    def higher_is_better(self):
        years = ["2024", "2023"]
        subjects = [
            "portuguese",
            "mathematics",
            "history",
            "physics",
            "chemistry",
            "geography",
            "biology",
            "english",
            "philosophy",
        ]

        years_agg_dict = {year: True for year in years}
        subjects_agg_dict = {subject: True for subject in subjects}

        return {
            "acc": True,
            "acc_norm": True,
            "BK": True,
            "IU": True,
            "USP": True,
            "UNICAMP": True,
            "has_associated_images": True,
            **years_agg_dict,
            **subjects_agg_dict,
        }

    def aggregation(self):
        years = ["2024", "2023"]
        subjects = [
            "portuguese",
            "mathematics",
            "history",
            "physics",
            "chemistry",
            "geography",
            "biology",
            "english",
            "philosophy",
        ]

        def safe_mean(values):
            if len(values) == 0:
                return -1
            return mean(values)
        
        years_agg_dict = {year: safe_mean for year in years}
        subjects_agg_dict = {subject: safe_mean for subject in subjects}

        
        return {
            "acc": safe_mean,
            "acc_norm": safe_mean,
            "BK": safe_mean,
            "IU": safe_mean,
            "USP": safe_mean,
            "UNICAMP": safe_mean,
            "has_associated_images": safe_mean,
            **years_agg_dict,
            **subjects_agg_dict,
        }
    
    @utils.positional_deprecated
    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None, conversation_template=None, prompt_as_single_user_message=False):
        """ Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param prompt_mode: str
            The type of prompt. Please set prompt_mode as "fixed", "dynamic-random", or "dynamic-similar".
            WARNING: this is implemented only for Portuguese tasks.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print("WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict")

        def adapt_text_to_conversation(text):
            # Remove '\nReponse: ', '\nSentiment: ', '\nScore:', etc. at the end of text
            if text[-1] == ':':
                text = text.rsplit('\n', 1)[0]
            return text
        
        
        def split_text_and_images(input_string, images):
            # Split the string using a regular expression that matches the image placeholders
            parts = re.split(r'(\[IMAGE \d+\])', input_string)
            # Filter out any empty strings from the list
            parts = [part for part in parts if part]
            

            processed_messages=[]

            for part in parts:
                if re.match(r'(\[IMAGE \d+\])', part):
                    # If the part is an image placeholder, replace it with the corresponding image
                    image_index= int(re.findall(r'\d+', part)[0])
                    
                    processed_messages.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{images[image_index]}",
                            },
                        }
                    )
                else:
                    processed_messages.append({"type": "text", "text": part})
                    
            return processed_messages
        
        if conversation_template:
            conversation = get_conv_template(conversation_template)
            user_role, assistant_role = conversation.roles
            assert description, "Conversation prompt requires a description."
        else:
            description = description + "\n\n" if description else ""

        example = self.doc_to_text(doc)

        if num_fewshot == 0:
            labeled_examples = ""
            if conversation_template:
                example = adapt_text_to_conversation(example)


                if self.mode in [BLUEX_EVAL_MODE.BLIND_CAPTION, BLUEX_EVAL_MODE.CONTEXT_CAPTION]:
                    # if we have description, use it. Replace the first placeholder with the description.
                    # descriptions for tables are ignored because the placeholder is added for images.
                    # experiment with_captions

                    if self.mode == BLUEX_EVAL_MODE.CONTEXT_CAPTION:
                        captions= doc['context_captions']
                    else:
                        captions= doc['blind_captions']

                    for i in range(len(captions)):
                        example= example.replace(f"[IMAGE {i}]", f"Descrição da imagem: {captions[i]}")
                    
                    conversation.append_message(user_role, description + "\n" + example)

                
                elif self.mode == BLUEX_EVAL_MODE.IMAGE:
                    # if we have placeholders and images, add the images in the prompt.
                    # experiment with_images
                    
                    contents = [{"type": "text", "text": description}, *split_text_and_images(example, doc['associated_images'])]
                    
                    conversation.append_message(user_role, contents)
                elif self.mode == BLUEX_EVAL_MODE.IGNORE_IMAGES:
                    # if we have placeholders, but no image, we remove the placeholders.
                    # it means the images were purposely excluded.
                    # experiment without_images
                    example = re.sub(r'\[IMAGE \d+\]', '', example)
                    conversation.append_message(user_role, description + "\n" + example)
                else:
                    conversation.append_message(user_role, description + "\n" + example)
                conversation.append_message(assistant_role, None)
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                # fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
                ## keeping the training docs in original order (use this to fixed prompts)
                fewshotex = list(self.training_docs())[:num_fewshot]
                ## if the current doc is among the training docs, we do not use it as few-shot
                fewshotex = [ex for ex in fewshotex if doc['id'] != ex['id']]
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs() if self.has_validation_docs() else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = ''
            
            if conversation_template:
                conversation.append_message(user_role, description)
                conversation.append_message(assistant_role, "Ok, vamos lá.")

                for i, doc_ex in enumerate(fewshotex):
                    text = adapt_text_to_conversation(self.doc_to_text(doc_ex))
                    target = self.doc_to_target(doc_ex).strip()
                    conversation.append_message(user_role, text)
                    conversation.append_message(assistant_role, target)
                
                example = adapt_text_to_conversation(example)
                
                if self.mode in [BLUEX_EVAL_MODE.BLIND_CAPTION, BLUEX_EVAL_MODE.CONTEXT_CAPTION]:
                    # if we have description, use it. Replace the first placeholder with the description.
                    # descriptions for tables are ignored because the placeholder is added for images.
                    # experiment with_captions

                    if self.mode == BLUEX_EVAL_MODE.CONTEXT_CAPTION:
                        captions= doc['context_captions']
                    else:
                        captions= doc['blind_captions']

                    for i in range(len(captions)):
                        example= example.replace(f"[IMAGE {i}]", f"Descrição da imagem: {captions[i]}")
                    
                    conversation.append_message(user_role, example)

                
                elif self.mode == BLUEX_EVAL_MODE.IMAGE:
                    # if we have placeholders and images, add the images in the prompt.
                    # experiment with_images
                    
                    contents = [ *split_text_and_images(example, doc['associated_images'])]
                    
                    conversation.append_message(user_role, contents)
                elif self.mode == BLUEX_EVAL_MODE.IGNORE_IMAGES:
                    # if we have placeholders, but no image, we remove the placeholders.
                    # it means the images were purposely excluded.
                    # experiment without_images
                    example = re.sub(r'\[IMAGE \d+\]', '', example)
                    conversation.append_message(user_role, example)
                else:
                    conversation.append_message(user_role, example)
                conversation.append_message(assistant_role, None)
            else:
                for i, doc_ex in enumerate(fewshotex):
                    labeled_examples += f'Questão {i+1}:\n'
                    labeled_examples += self.doc_to_text(doc_ex) + self.doc_to_target(doc_ex)
                    labeled_examples += '\n##\n'
                labeled_examples += f'Questão {len(fewshotex) + 1}:\n'

        if conversation_template:
            if prompt_as_single_user_message:
                return conversation.get_prompt()
            else:
                return json.dumps(conversation.to_openai_api_messages(), ensure_ascii=False)
        else:
            return description + labeled_examples + example
        

class BLUEX_BLIND_CAPTIONS(BLUEX):
    COT = False
    mode = BLUEX_EVAL_MODE.BLIND_CAPTION

class BLUEX_CONTEXT_CAPTIONS(BLUEX):
    COT = False
    mode = BLUEX_EVAL_MODE.CONTEXT_CAPTION

class BLUEX_IMAGES(BLUEX):
    COT = False
    mode = BLUEX_EVAL_MODE.IMAGE

class BLUEX_IGNORE_IMAGES(BLUEX):
    COT = False
    mode = BLUEX_EVAL_MODE.IGNORE_IMAGES

class BLUEX_BLIND_CAPTIONS_COT(BLUEX):
    COT = True
    mode = BLUEX_EVAL_MODE.BLIND_CAPTION

class BLUEX_CONTEXT_CAPTIONS_COT(BLUEX):
    COT = True
    mode = BLUEX_EVAL_MODE.CONTEXT_CAPTION
    
class BLUEX_IMAGES_COT(BLUEX):
    COT = True
    mode = BLUEX_EVAL_MODE.IMAGE

class BLUEX_IGNORE_IMAGES_COT(BLUEX):
    COT = True
    mode = BLUEX_EVAL_MODE.IGNORE_IMAGES