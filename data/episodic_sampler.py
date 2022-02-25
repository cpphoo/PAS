import numpy as np
import torch

from torch.utils.data import DataLoader
import copy 


def generate_class_to_id_map(dataset):
    '''
        Generate a map where
            - key is the class id
            - value is a list of index of examples from class id
    '''
    targets = dataset.targets[:, -1]

    targets = targets.cpu().tolist()

    class_to_id = {}

    for ind, i in enumerate(targets):
        if i not in class_to_id:
            class_to_id[i] = []
        
        class_to_id[i].append(ind)
    
    for i in class_to_id:
        class_to_id[i] = np.array(class_to_id[i])

    return class_to_id

class Episodic_wrapper:
    '''
        A generic wrapper on ImageFolder used to generates episodes
        - ImageFolder should have
            - target - an (N, ) label tensor where N is the number of examples 
            - __getitem__(self, idx) - getter function to retrieve specific examples and labels
        
        Args:
            dataset: an instance of ImageFolder
            num way: Number of classes in an episode
            num_ref: number of reference/support examples per class in an episode
            num_query: number of query examples per class in an episode
            num_episodes: number of episodes in an epoch
    '''
    def __init__(self, dataset, num_way=5, num_ref=5, num_query=15, num_episodes=1000):
        self.dataset = dataset
        self.num_way = num_way
        self.num_ref = num_ref
        self.num_query = num_query
        self.num_episodes = num_episodes
        
        self.class_to_id = generate_class_to_id_map(dataset)

        self.base_classes = np.array(list(self.class_to_id.keys()))

        # generate episodes 
        self.episodes = []

        self._generate_episodes()
        return 
        

    def __getitem__(self, index, relabel=False):
        '''
        Args:
            index (int): Index
            relabel (bool): whether to relabel the examples in an episode to [num_way - 1]

        Returns:
            An episode of the form (Xref, yref, Xquery, yquery)

        '''

        id_ref, id_query, episode_classes = self.episodes[index]

        ref = [self.dataset[i] for i in id_ref]
        query = [self.dataset[i] for i in id_query]

        Xref = torch.stack([r[0] for r in ref])
        yref = torch.stack([r[1] for r in ref])

        Xquery = torch.stack([q[0] for q in query])
        yquery = torch.stack([q[1] for q in query]) 

        if relabel: 
            for ind, i in enumerate(episode_classes):
                yref[yref == i] = ind
                yquery[yquery == i] = ind

        return Xref, yref, Xquery, yquery

    def generate_loader(self, **kwargs):
        # kwargs are the arugment to dataloader
        # Generate new list of episodes 
        # and return a loader 
        # this is a convenient function to avoid potential bugs
        # i.e. not generating new episodes when go into a new epoch
        self._generate_episodes()
        return DataLoader(self, **kwargs)

    def _generate_episodes(self):
        episodes = []
        tracker_leftover = {c: np.array([True] * len(self.class_to_id[c])) for c in self.class_to_id}
        num_examples_per_class = self.num_ref + self.num_query

        for _ in range(self.num_episodes):

            id_ref = []
            id_query = []

            # first sample classes
            classes_episode = np.random.choice(self.base_classes, size=self.num_way, replace=False)
            # Then sample examples per class 

            for c in classes_episode:
                ind_leftover = np.where(tracker_leftover[c])[0]

                if len(ind_leftover) < num_examples_per_class:
                    to_be_added = copy.deepcopy(ind_leftover)

                    tracker_leftover[c] = np.array([True] * len(self.class_to_id[c]))
                    tracker_leftover[c][to_be_added] = False

                    ind_leftover = np.where(tracker_leftover[c])[0]

                    ind_additional = np.random.choice(ind_leftover, size=num_examples_per_class - len(to_be_added), replace=False)
                    
                    to_be_added = np.concatenate([to_be_added, ind_additional])
                else:
                    to_be_added = np.random.choice(ind_leftover, size=num_examples_per_class, replace=False)

                id_ref += self.class_to_id[c][to_be_added[:self.num_ref]].tolist()
                id_query += self.class_to_id[c][to_be_added[self.num_ref:]].tolist()

            np.random.shuffle(id_ref)
            np.random.shuffle(id_query)
            episodes.append([id_ref, id_query, classes_episode.tolist()])

        self.episodes = episodes
        assert len(self.episodes) == self.num_episodes 
        return

    def __len__(self):
        return self.num_episodes



class Episodic_wrapper_parent(Episodic_wrapper):
    def __init__(self, dataset, num_way=5, num_ref=5, num_query=15, num_episodes=1000):
        parent_to_children_map = {}

        for i in dataset.lvl_child[-1]:
            parent_to_children_map[dataset.class_to_idx_hierarchy[-2][i]] = np.array([
                dataset.class_to_idx_hierarchy[-1][j] for j in dataset.lvl_child[-1][i]])

        # A mapping that maps parent class id to its children id
        # parent_to_children_map[parent_id] --> a list of children id
        self.parent_to_children_map = parent_to_children_map
        self.parent_classes = list(parent_to_children_map.keys())

        # the maximum number of parent per episode, this is to ensure that for each parent
        # there will be at least two children in an episode
        self.num_parent_max = int(
            min(np.floor(num_way / 2), len(self.parent_to_children_map)))
        super().__init__(dataset, num_way, num_ref, num_query, num_episodes)

        

    def _generate_episodes(self):
        episodes = []
        tracker_leftover = {c: np.array(
            [True] * len(self.class_to_id[c])) for c in self.class_to_id}
        num_examples_per_class = self.num_ref + self.num_query

        for _ in range(self.num_episodes):

            id_ref = []
            id_query = []

            # sample parent classes
            # decide parent classes
            num_parent_classes = np.random.randint(low=2, high=self.num_parent_max + 1, size=(1, ))
            
            # Need to make sure that the number of children can form an episode
            inconsistent = True
            while inconsistent:
                parent_class_episode = np.random.choice(self.parent_classes, size=num_parent_classes, replace=False).tolist()

                count = 0
                for i in parent_class_episode:
                    count += len(self.parent_to_children_map[i])

                if count >= self.num_way:
                    inconsistent = False
                else:
                    inconsistent = True

            # Use to keep track of leftover children for each parent
            children_leftover = {i: np.array([True] * len(self.parent_to_children_map[i])) for i in parent_class_episode}

            classes_episode = []

            for p in parent_class_episode:
                num_children = len(self.parent_to_children_map[p])
                # to ensure that each parent appeared twice
                ind_to_be_added = np.random.permutation(num_children)[:2]
                children_leftover[p][ind_to_be_added] = False
                classes_episode += [self.parent_to_children_map[p][j] for j in ind_to_be_added]

            leftover_children = [self.parent_to_children_map[p][children_leftover[p]] for p in parent_class_episode]
            leftover_children = np.concatenate(leftover_children)

            classes_episode += np.random.choice(leftover_children, size=self.num_way - num_parent_classes * 2, replace=True).tolist()

            classes_episode = np.array(classes_episode)

            assert len(classes_episode) == self.num_way
            # Then sample examples per class

            for c in classes_episode:
                ind_leftover = np.where(tracker_leftover[c])[0]

                if len(ind_leftover) < num_examples_per_class:
                    to_be_added = copy.deepcopy(ind_leftover)

                    tracker_leftover[c] = np.array(
                        [True] * len(self.class_to_id[c]))
                    tracker_leftover[c][to_be_added] = False

                    ind_leftover = np.where(tracker_leftover[c])[0]

                    ind_additional = np.random.choice(
                        ind_leftover, size=num_examples_per_class - len(to_be_added), replace=False)

                    to_be_added = np.concatenate([to_be_added, ind_additional])
                else:
                    to_be_added = np.random.choice(
                        ind_leftover, size=num_examples_per_class, replace=False)

                id_ref += self.class_to_id[c][to_be_added[:self.num_ref]].tolist()
                id_query += self.class_to_id[c][to_be_added[self.num_ref:]].tolist()

            np.random.shuffle(id_ref)
            np.random.shuffle(id_query)
            episodes.append([id_ref, id_query, classes_episode.tolist()])

        self.episodes = episodes
        assert len(self.episodes) == self.num_episodes
        return 

if __name__ == "__main__":
    from ImageFolder import ImageFolder
    import time

    import torchvision

    dataset = dataset = ImageFolder(
        '/media/cheng/Samsung_T5/inat/hierarchical_meta_inat/repr/all_no_superclass',
        label_file='/media/cheng/Samsung_T5/inat/hierarchical_meta_inat/aves_base.csv',
        transform=torchvision.transforms.ToTensor())

    start = time.time()
    dataset_episodic = Episodic_wrapper_parent(dataset, num_way=5)
    end = time.time()
    print('Time Elapsed: ', end - start)
    
    start = time.time()
    Xref, yref, Xquery, yquery = dataset_episodic[0]
    end = time.time()
    print('Time Elapsed: ', end - start)

    print(Xref.shape)
    print(yref.shape)
    print(torch.unique(yref[:, -1]))
    print(torch.unique(yref[:, -2]))

    print(Xquery.shape)
    print(yquery.shape)
    print(torch.unique(yquery[:, -1]))
    print(torch.unique(yquery[:, -2]))

    print(len(dataset_episodic))

    start = time.time()
    loader = dataset_episodic.generate_loader(batch_size=8, shuffle=True, num_workers=4)
    end = time.time()
    print('Time Elapsed: ', end - start)

    Xref, yref, Xquery, yquery = iter(loader).next()

    print(Xref.shape)
    print(yref.shape)

    print(torch.unique(yref[0, :, -1]))
    print(torch.unique(yref[1, :, -1]))
