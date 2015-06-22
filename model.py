# -*- coding: utf-8 -*-



class Document(object):
    """
    models document object
    """
    def __init__(self, **kwargs):
        self.lang = kwargs.pop('language', "en")
        self.name =kwargs.pop('name','')
        self.documents = kwargs.pop("document", [])
        self.author = kwargs.pop('author', Author())


    def add_document(self, doc):
        self.documents.append(doc)

    @property
    def content(self):
        return " ".join(self.documents)



class PersonalityTraits(object):
    """
    models personality traits
    """
    def __init__(self, **kwargs):
        self.extroverted = kwargs.get("extroverted", 0.0)
        self.stable = kwargs.get("stable", 0.0)
        self.agreeable = kwargs.get("agreeable", 0.0)
        self.conscientious = kwargs.get("conscientious", 0.0)
        self.open = kwargs.get("open", 0.0)


class Author(object):
    """
    models author
    """
    def __init__(self, **kwargs):
        self._gender = kwargs.get('gender', '')
        self._age_group = kwargs.get('age_group', '')
        self.personality_traits = kwargs.get('personality_traits', PersonalityTraits())

    @property
    def gender(self):
        if self._gender == 'M' or self._gender == 'm':
            return  'male'
        else:
            return 'female'

    @gender.setter
    def gender(self,value):
        self._gender=value

    @property
    def age_group(self):
        return self._age_group.lower()

    @age_group.setter
    def age_group(self,value):
        self._age_group=value

    @property
    def extroverted(self):
        return str(round(self.personality_traits.extroverted,3))


    @property
    def stable(self):
        return str(round(self.personality_traits.stable,3))


    @property
    def agreeable(self):
        return str(round(self.personality_traits.agreeable,3))

    @property
    def conscientious(self):
        return str(round(self.personality_traits.conscientious,3))

    @property
    def open(self):
        return str(round(self.personality_traits.open,3))