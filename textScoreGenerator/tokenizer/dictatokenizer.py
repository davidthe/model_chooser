# coding=utf-8
from ctypes import ArgumentError
import json
import os
from transformers import BasicTokenizer, BertTokenizer
from transformers.tokenization_utils import _is_punctuation

ALL_MODERN_HEBREW_PREFIXES = set(["ב", "בכ", "בש", "בשה", "בשל", "ה", "ו", "כ", "כב", "כש", "כשב", "כשה", "כשכ", "כשל", "כשמ", "כשמה", "ל", "לכ", "לכש", "לכשה", "למ", "למב", "למש", "לש", "לשל", "מ", "מב", "מבש", "מה", "מכ", "מכש", "מל", "מלש", "מש", "משה", "משל", "ש", "שב", "שה", "שכ", "שכש", "שכשה", "של", "שלכש", "שלכשה", "שלש", "שמ", "שמב", "שמה", "שמש", "וב", "ובכ", "ובש", "ובשה", "ובשל", "וה", "וכ", "וכב", "וכש", "וכשב", "וכשה", "וכשכ", "וכשל", "וכשמ", "וכשמה", "ול", "ולכ", "ולכש", "ולכשה", "ולמ", "ולמב", "ולמש", "ולש", "ולשל", "ומ", "ומב", "ומבש", "ומה", "ומכ", "ומכש", "ומל", "ומלש", "ומש", "ומשה", "ומשל", "וש", "ושב", "ושה", "ושכ", "ושכש", "ושכשה", "ושל", "ושלכש", "ושלכשה", "ושלש", "ושמ", "ושמב", "ושמה", "ושמש"])
ALL_MODERN_HEBREW_APOS_WORDS = set(["פרופ", "ג'ורג", "עמ", "א", "ר", "ה", "ב", "מס", "מ", "הפרופ", "ג", "נק", "ד", "לינץ", "יחימוביץ", "ושות", "ברקוביץ", "סמית", "י", "ופרופ", "אינץ", "ו", "טל", "ט", "ש", "קולג", "ח", "נ", "אלוביץ", "אברמוביץ", "ז", "קוטג", "וכו", "רבינוביץ", "בקולג", "דוידוביץ", "ריב", "ז'ורז", "חג'ג", "אורנג", "הרציקוביץ", "רח", "ל", "במיל", "ס", "פייג", "ק", "פרופ", "לפרופ", "וויצ'יץ", "וג'ורג", "בעמ", "קיימברידג", "ארביטראז", "גורביץ", "בורוביץ", "הקוטג", "ביץ", "פיץ", "פ", "שייח", "ע", "בטל", "מילושביץ", "הברוקראז", "ליבוביץ", "חיימוביץ", "ברידג", "חאג", "לר", "ריץ", "סאות", "לה", "קית", "אהרונוביץ", "גב", "פורטסמות", "הקולג", "מרקוביץ", "כ", "השייח", "ברומיץ", "ג'וקוביץ", "ור", "אסאנג", "שד", "פופוביץ", "בז", "הארץ", "אופ", "טאץ", "לא", "ברח", "סנדוויץ", "ריסרץ", "סמוטריץ", "ברוקראז", "ינוקוביץ", "איברהימוביץ", "קייג", "לג'ורג", "דק", "וינטג", "קומטאץ", "הבעת", "והפרופ", "קולאז", "בה", "וילג", "נורת", "גינגריץ", "הגב", "אס", "לייבוביץ", "שפרופ", "פוניבז", "ברוז", "סרג", "רוז", "מוסקוביץ", "דיוקוביץ", "דר", "מיראז", "הלינץ", "גארת", "שוסטקוביץ", "הפרינג", "סרז", "קראנץ", "בקיימברידג", "ישראלביץ", "אימאג", "ליאז", "מלובביץ", "לודז", "איליץ", "פרנץ", "צ", "רות", "הפ", "סת", "פריג", "פרג", "אג", "מיץ", "לקולג", "קנת", "שג'ורג", "טק-קראנץ", "שיח", "בת", "טורבוביץ", "בלוטות", "אליזבת", "בא", "אייג", "דנילוביץ", "בורגראנץ", "איוונוביץ", "יחימוביץ", "סטויאקוביץ", "וה", "ת", "שר", "אסיס", "וב", "אובראדוביץ", "אימג", "אינצ", "שה", "פרינג", "בר", "ראג", "גריניץ", "לואץ", "בלודז", "בד", "הומאז", "דודג", "איצקוביץ", "בט", "הגראנג", "איפסוויץ", "בורג", "הארביטראז", "קלאץ", "אלכסנדרוביץ", "דת", "ינקוביץ", "יעקובוביץ", "שנ", "קראוץ", "סטאז", "זבלודוביץ", "לבקוביץ", "א-שייח", "מש", "הסנדוויץ", "קובאץ", "גר", "פטרוביץ", "בג", "סי", "מפרופ", "בוגדנוביץ", "בי", "וגו", "הלת", "בגריניץ", "קוראץ", "תר", "מסאז", "דאץ", "קילומטראז", "מלאדיץ", "הסטאז", "ראנץ", "נוריץ", "אדג", "מודריץ", "בח", "רואץ", "אוברקוביץ", "בראנץ", "בכ", "שא", "מא", "הץ", "וג", "אלדריג", "טאג", "רידג", "דארת", "מרטינוביץ", "שיחימוביץ", "קובוביץ", "קלוז", "השיח", "ביליץ", "דוקיץ", "קוץ", "לאלוביץ", "גריפית", "איוואנוביץ", "בלסאות", "וונאג", "טוב", "בפרופ", "באורנג", "ומס", "יבריץ", "הית", "ליחימוביץ", "מינאז", "קלמנוביץ", "ליוביצ'יץ", "סגלוביץ", "ויטראז", "מארג", "באריץ", "מרדית", "הנק", "באת", "דווידוביץ", "ו-ב", "פינץ", "גדז", "שלי", "בב", "ממשפ", "בו", "יאנקוביץ", "וא", "אינג", "בלינץ", "צ'יליץ", "דראגיץ", "אוברדוביץ", "פיית", "סוויץ", "איבקוביץ", "חג", "קולרידג", "מירוסביץ", "ג'ינג'יץ", "מרג", "וו", "צ'רץ", "נצ", "פס", "מנדלוביץ", "זליקוביץ", "הרשקוביץ", "יקירביץ", "הטאג", "הבז", "דנץ", "ניקוליץ", "שברקוביץ", "וקית", "מהקולג", "גורג", "המיראז", "גדג", "וידיץ", "בות", "בלייג", "הטאץ", "אנילביץ", "סטייג", "וד", "ואח", "לברקוביץ", "בקוטג", "איבנוביץ", "באוני", "גונטז", "וינטאג", "לודג", "מנ", "וי", "סטארידג", "מר", "מיכאילוביץ", "מקבת", "ריינג", "אקסצ'יינג", "מקיימברידג", "וברקוביץ", "ידיעות", "מג'ורג", "ראיקוביץ", "הברידג", "בוץ", "כפרופ", "הדת", "לגראנז", "מאריץ", "וולפוביץ", "קילינגסוורת", "ויחימוביץ", "זבלדוביץ", "ו-ג", "שמואלביץ", "מלקוביץ", "דמירוביץ", "שאלוביץ", "בשייח", "טאצ", "הוקיץ", "רביקוביץ", "טאדיץ", "וודג", "פ.ס.ז", "הקולאז", "מרקטווץ", "מירג", "פאנץ", "מינטקביץ", "הקילומטראז", "שו", "טומיץ", "סטריינג", "קרלוביץ", "קארדז'יץ", "הריטג", "באונ", "קילומטרז", "החאג", "בפורטסמות", "אנטיץ", "גראנג", "ואורנג", "אנ", "ברקוביץ", "מיל", "וז'ורז", "נאג", "טרנץ", "אוליץ", "לאקוביץ", "הקלאץ", "צ'יץ", "בוסקוביץ", "מאץ", "ראוניץ", "לקיימברידג", "לַה", "מארץ", "לייז", "הביץ", "לב", "אייץ", "קובאצ'וויץ", "קולידג", "מורגות", "ראדוביץ", "החדית", "לגב", "אל-חאג", "אונ", "וידוביץ", "הבלוטות", "האץ", "מיטרוביץ", "קז'ימייז", "ליץ", "סטונהנג", "ופייג", "פלאניניץ", "הסוויץ", "סאריץ", "כב", "וורת", "וסמית", "אנצ'יץ", "שמ", "לימוז", "דראפיץ", "צ'ורצ'יץ", "גודוביץ", "פסחוביץ", "בוליץ", "בוצ'אץ", "איוואניסביץ", "להרציקוביץ", "וכד", "פאדג", "בג'ורג", "פראג", "גלט-ברקוביץ", "פירת", "ורסאץ", "איזיקוביץ", "קוראז", "רייג", "ג'אדג", "איינג", "לאונג", "פלימות", "סרגייביץ", "לאברמוביץ", "אולדריץ", "ה\"ביץ", "ספיץ", "ניקולאייביץ", "מירוטיץ", "קוקוץ", "פרת", "בל", "לפורטסמות", "פקוביץ", "לקוטג", "גז'גוז", "סאביץ", "לאורנג", "ענח", "מחיר", "שאברמוביץ", "פטקוביץ", "הפרופ", "לג", "מת", "ולדימירוביץ", "ברדיץ", "הניו-אייג", "קואץ", "סייג", "גריגורייביץ", "מונטאז", "חב", "הווינטג", "סקרוג", "מוזס-בורוביץ", "בורנמות", "הברוקרג", "וח", "לד", "יב", "צינוביץ", "בקולז", "קוסטיץ", "יובאנוביץ", "גייג", "ללינץ", "מד", "נייסמית", "ופיץ", "דונביץ", "פץ", "טלטוביץ", "מרינקוביץ", "ולפרופ", "ברלוביץ", "משקביץ", "לוביץ", "רוסטרופוביץ", "בריץ", "טיפסרביץ", "קנטרידג", "וע", "כשג'ורג", "וש", "גווינת", "ו-ה", "ואלוביץ", "שעת", "בולאטוביץ", "גורוז", "כר", "ברנוביץ", "פונץ", "סטנויביץ", "מקולג", "מיאטוביץ", "זאהוביץ", "קוטג", "מי", "קרסטיץ", "מדז'יבוז", "בפיץ", "לארג", "טומאסיץ", "בראטיץ", "ראקיטיץ", "בנטוויץ", "בורז", "סות", "הודג", "שהרציקוביץ", "בווילג", "אנבת", "גודריץ", "גארבג", "פסאז", "וסרג", "לש", "הר", "אל-בורייג", "ארת", "לובץ", "בקולאז", "למ", "בפרת", "מיליצ'יץ", "זיסוביץ", "מילוטינוביץ", "צ'וברביץ", "שסטוביץ", "סולאראדג", "במס", "מייברידג", "יוקיץ", "מאירוביץ", "קמברבאץ", "הגראז", "אינטרנט", "מיכאלוביץ", "פאביצ'ביץ", "סטוראייג", "נימצוביץ", "סטאנקוביץ", "לשייח", "ג'רג", "לדוידוביץ", "מצקביץ", "לובביץ", "יח", "ארמיטאג", "מח", "בירץ", "ובז", "חדית", "סע", "כשפרופ", "יוליץ", "מאליקוביץ", "ובקולג", "בסאות", "שדוידוביץ", "בגראז", "מאטוסביץ", "סנדביץ", "בליאז", "הוצ", "ומ", "סבאת", "אניצ'יץ", "שהפרופ", "הבראנץ", "לוקאץ", "גראז", "פינקוביץ", "בצ'ירוביץ", "סיראז", "ספורטאז", "הקילומטרז", "שסמית", "בוריסוביץ", "מסיץ", "שמילושביץ", "בטק-קראנץ", "וורונז", "וודברידג", "יות", "האורנג", "אוני", "קרנץ", "הוויטראז", "פלאת", "מגאת", "לאברינוביץ", "וט", "רוץ", "בקלוז", "הפסאז", "הפאנץ"])

class OurBasicTokenizer(BasicTokenizer):
    do_rabbinic_quotes_break = False
    do_modern_quotes_break = False
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if text in self.never_split or (never_split and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        is_unk_word = False
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_unk_char(char):
                if not is_unk_word:
                    output.append(list('[UNK]'))
                #output[-1].append(char)
                is_unk_word = True
            else:
                extra_quotes_break = False
                if (char == '"' or char == "'") and not start_new_word and (self.do_modern_quotes_break or self.do_rabbinic_quotes_break) and _is_hebrew_let(output[-1][-1]):
                    if self.do_rabbinic_quotes_break:
                        extra_quotes_break = char == "'" or (char == '"' and i + 1 < len(chars) and _is_hebrew_let(chars[i + 1]))
                    if self.do_modern_quotes_break:
                        # explanation: If it's an apostrophe completing a valid apostrophe word - keep
                        # if it's a double quote followed by a letter - keep it, unless the letters so far are 
                        # a valid prefix and the next 3 are letters
                        extra_quotes_break = (char == "'" and ''.join(output[-1]) in ALL_MODERN_HEBREW_APOS_WORDS) or \
                                                (char == '"' and i + 1 < len(chars) and _is_hebrew_let(chars[i + 1]) and not (i + 3 < len(chars) and _is_hebrew_let(chars[i + 2]) and _is_hebrew_let(chars[i + 3]) and ''.join(output[-1]) in ALL_MODERN_HEBREW_PREFIXES))

                if _is_punctuation(char) and not extra_quotes_break:
                    output.append([char])
                    start_new_word = True
                    is_unk_word = False
                else:
                    if start_new_word or is_unk_word:
                        output.append([])
                    start_new_word = False
                    is_unk_word = False
                    output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

def _is_unk_char(char):
    cp = ord(char)
    # I think we can do this to any character not in the following code blocks: 
    # Anything in the Hebrew block (0x0590-0x05FF); 
    if cp >= 0x590 and cp <= 0x05FF: return False
    # Anything in the “Basic Latin” (0x0-0x7F); 
    if cp >= 0x0 and cp <= 0x7F: return False
    # Anything in the “General Punctuation” block (200C-203f); 
    if cp >= 0x200C and cp <= 0x203F: return False
    # Currency Symbols (20A0-20BF); 
    if cp >= 0x20A0 and cp <= 0x20BF: return False
    # Mathematical Operators (2200-22FF); 
    if cp >= 0x2200 and cp <= 0x22FF: return False
    # Number Forms (2150-218B); 
    if cp >= 0x2150 and cp <= 0x218B: return False
    # AlphabeticPresentationForms (FB00-FB4F)
    if cp >= 0xFB00 and cp <= 0xFB4F: return False
    return True

def _is_hebrew_let(char):
    cp = ord(char)
    return cp >= 0x5D0 and cp <= 0x5EA

def DictaBertRabbinicTokenizer(tok):
    tok.basic_tokenizer = OurBasicTokenizer(tok.basic_tokenizer.do_lower_case, tok.basic_tokenizer.never_split)
    tok.basic_tokenizer.do_rabbinic_quotes_break = True
    return tok

def DictaBertNewModernTokenizer(tok):
    tok.basic_tokenizer = OurBasicTokenizer(tok.basic_tokenizer.do_lower_case, tok.basic_tokenizer.never_split)
    tok.basic_tokenizer.do_modern_quotes_break = True
    return tok

def DictaBertTokenizer(tok):
    tok.basic_tokenizer = OurBasicTokenizer(tok.basic_tokenizer.do_lower_case, tok.basic_tokenizer.never_split)
    return tok


class DictaAutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the [`AutoTokenizer.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "DictaAutoTokenizer is designed to be instantiated "
            "using the `DictaAutoTokenizer.from_pretrained(pretrained_model_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, *inputs, **kwargs) -> BertTokenizer:
        r"""
        Instantiate the BertTokenizer with the Dicta wrapper using the properties of the config object (loaded from `pretrained_model_path` if possible)

        List options

        Params:
            pretrained_model_path (`str` or `os.PathLike`):
                A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__()` method.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the Tokenizer `__init__()` method. Can be used to set special tokens like
                `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
                `additional_special_tokens`. See parameters in the `__init__()` for more details.

        Examples:

        ```python
        # >>> from dictatokenizer import DictaAutoTokenizer
        # >>> tokenizer = DictaAutoTokenizer.from_pretrained("BerelSefaria_34800")
        ```"""
        
        if not os.path.isdir(pretrained_model_path):
            if os.path.isfile(pretrained_model_path):
                raise ArgumentError(
                    "DictaAutoTokenizer only accepts a path to a directory containing the pretrained model, "
                    "not to a specific file"
                )
            else:
                raise FileNotFoundError(
                    "Path given for model doesn't exist"
                )

        config_fname = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_fname):
            raise FileNotFoundError(
                "Config file for given model doesn't exist"
            )

        # load the tokenizer
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_path, *inputs, **kwargs)

        # load the config and do the wrapping
        with open(config_fname, 'r') as r:
            config = json.loads(r.read())

        if 'rabbinic' in config and config['rabbinic']:
            tokenizer = DictaBertRabbinicTokenizer(tokenizer)
        elif 'newmodern' in config and config['newmodern']:
            tokenizer = DictaBertNewModernTokenizer(tokenizer)
        else:
            tokenizer = DictaBertTokenizer(tokenizer)

        return tokenizer
