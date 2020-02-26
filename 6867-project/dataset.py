class DatasetContainer:
    """
    Represents a container object for a dataset

    Fields:
    - ids: list of document IDs contained within the dataset
    - dups: list of duplicate post entries, where each entry is [post1, post2, dup]
    - bodies: list of document bodies contained within the dataset
    - titles: list of document titles contained within the dataset
    - combined: list of body+title concatenated contained with the dataset
    - tf_body: term-document matrix for post bodies
    - tf_title: term-document matrix for post titles
    - tf_combined: term-document matrix for combined body + title
    - X: the feature (data) matrix 
    - Y: the vector of labels
    - reputations
    - scores

    """
    def __init__(self, ids, dups, bodies, titles, combined, tf_body, tf_title, tf_combined, reputations, scores):
        self.ids = ids
        self.dups = dups
        self.scores = scores
        self.bodies = bodies
        self.titles = titles
        self.combined = combined
        self.tf_body = tf_body
        self.tf_title = tf_title
        self.tf_combined = tf_combined
        self.reputations = reputations
        self.X = None
        self.Y = None
