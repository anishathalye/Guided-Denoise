import robustml
from model_robustml import Model
import torch
import torch.autograd as autograd
import sys
import argparse
import numpy as np

class PGD(robustml.attack.Attack):
    def __init__(self, model, epsilon, learning_rate=1.0/255.0, max_steps=100, debug=False):
        self._model = model
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._max_steps = max_steps
        self._debug = debug
        self._xent = torch.nn.CrossEntropyLoss()

    def run(self, x, y, target):
        if target is not None:
            raise NotImplementedError
        adv = np.transpose(np.array([x]), (0,3,1,2))
        lower = np.clip(adv - self._epsilon, 0, 1)
        upper = np.clip(adv + self._epsilon, 0, 1)
        target = autograd.Variable(torch.LongTensor(np.array([y])).cuda())
        for i in range(self._max_steps):
            input_var = autograd.Variable(torch.FloatTensor(adv).cuda(), requires_grad=True)
            input_tf = (input_var - self._model._mean_tf) / self._model._std_tf
            input_torch = (input_var - self._model._mean_torch) / self._model._std_torch
            
            labels1 = self._model._net1(input_torch,True)[-1]
            labels2 = self._model._net2(input_tf,True)[-1]
            labels3 = self._model._net3(input_tf,True)[-1]
            labels4 = self._model._net4(input_torch,True)[-1]

            labels = labels1 + labels2 + labels3 + labels4
            p = labels.max(1)[1].data.cpu().numpy()[0]
            loss = self._xent(labels, target)

            if self._debug:
                print(
                    'attack: step %d/%d, loss = %g (true %d, predicted %d)' % (
                        i+1,
                        self._max_steps,
                        loss,
                        y,
                        p,
                    ),
                    file=sys.stderr
                )
            if y != p:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                break
            loss.backward()
            g = input_var.grad.data.cpu().numpy()
            adv += self._learning_rate * np.sign(g)
            adv = np.clip(adv, lower, upper)

        return np.transpose(adv, (0,2,3,1))[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', type=str, required=True,
            help='directory containing `val.txt` and `val/` folder')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # initialize a model
    model = Model()

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    attack = PGD(model, epsilon=model.threat_model.epsilon, debug=args.debug)

    # initialize a data provider for ImageNet images
    provider = robustml.provider.ImageNet(args.imagenet_path, model.dataset.shape)

    success_rate = robustml.evaluate.evaluate(
        model,
        attack,
        provider,
        start=args.start,
        end=args.end,
        deterministic=True,
        debug=args.debug,
    )

    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
