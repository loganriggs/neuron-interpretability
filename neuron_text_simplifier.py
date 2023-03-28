import torch as th
from circuitsvis.activations import text_neuron_activations
from jaxtyping import Float, Int
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)
from einops import rearrange
class NeuronTextSimplifier:
    def __init__(self, model, layer: int, neuron: int) -> None:
        self.model = model
        self.device = model.cfg.device
        self.layer = layer
        self.neuron = neuron
        self.model.requires_grad_(False)
        self.embed_weights = list(list(model.children())[0].parameters())[0]
        if("pythia" not in model.cfg.model_name):
            transformer_block_loc = 4
        else:
            transformer_block_loc = 2
        transformer_blocks = [mod for mod in list(self.model.children())[transformer_block_loc]]
        self.model_no_embed = th.nn.Sequential(*(transformer_blocks[:layer+1]))
        self.model_no_embed.requires_grad_(False)
        self.set_hooks()

    def set_hooks(self):
        self._neurons = th.empty(0)
        def hook(model, input, output):
            self._neurons = output
        self.model.blocks[self.layer].mlp.hook_post.register_forward_hook(hook)

    def ablate_mlp_neurons(self, tokens, neurons: th.Tensor):
        def mlp_ablation_hook(
            value: Float[th.Tensor, "batch pos d_mlp"],
            hook: HookPoint
        ) -> Float[th.Tensor, "batch pos d_mlp"]:
            if(neurons.shape[0] == 0):
                return value
            value[:, :, neurons] = 0
            return value
        return self.model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{self.layer}.mlp.hook_post", mlp_ablation_hook)])
        
    def add_noise_to_text(self, text, noise_level=1.0):
        if isinstance(text, str):
            text = [text]
        text_list = []
        activation_list = []
        for t in text:
            split_text = self.model.to_str_tokens(t, prepend_bos=False)
            tokens = self.model.to_tokens(t, prepend_bos=False)
            # Add gaussian noise to the input of each word in turn, getting the diff in final neuron's response
            embedded_tokens = self.model.embed(tokens)
            batch_size, seq_size, embedding_size = embedded_tokens.shape
            noise = th.randn(1, embedding_size, device=self.device)*noise_level
            original = self.embedded_forward(embedded_tokens)[:,-1,self.neuron]
            changed_activations = th.zeros(seq_size, device=self.device)
            for i in range(seq_size):
                embedded_tokens[:,i,:] += noise
                neuron_response = self.embedded_forward(embedded_tokens)
                changed_activations[i] = neuron_response[:,-1,self.neuron].item()
                embedded_tokens[:,i,:] -= noise
            changed_activations -= original
            text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
            activation_list += changed_activations.tolist() + [0.0]
        activation_list = th.tensor(activation_list).reshape(-1,1,1)
        return text_neuron_activations(tokens=text_list, activations=activation_list)

    def visualize_logit_diff(self, text, neurons: th.Tensor, setting="true_tokens", verbose=True):
        if isinstance(text, str):
            text = [text]
        text_list = []
        logit_list = []
        for t in text:
            split_text = self.model.to_str_tokens(t, prepend_bos=False)
            tokens = self.model.to_tokens(t, prepend_bos=False)
            original_logits = self.model(tokens).log_softmax(-1)
            ablated_logits = self.ablate_mlp_neurons(tokens, neurons).log_softmax(-1)
            diff_logits =  ablated_logits - original_logits
            if setting == "true_tokens":
                # Gather the logits for the true tokens
                diff = rearrange(diff_logits.gather(2,tokens.unsqueeze(2)), "b s n -> (b s n)")
            elif setting == "max":
                val, ind = diff_logits.max(2)
                diff = rearrange(val, "b s -> (b s)")
                split_text = self.model.to_str_tokens(ind)
                tokens = ind
            if(verbose):
                text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
                text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
                orig = rearrange(original_logits.gather(2,tokens.unsqueeze(2)), "b s n -> (b s n)")
                ablated = rearrange(ablated_logits.gather(2,tokens.unsqueeze(2)), "b s n -> (b s n)")
                logit_list += orig.tolist() + [0.0]
                logit_list += ablated.tolist() + [0.0]
            text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
            logit_list += diff.tolist() + [0.0]
        logit_list = th.tensor(logit_list).reshape(-1,1,1)
        if verbose:
            print(f"Max & Min logit-diff: {logit_list.max().item():.2f} & {logit_list.min().item():.2f}")
        return text_neuron_activations(tokens=text_list, activations=logit_list)

    def get_neuron_activation(self, tokens):
        _, cache = self.model.run_with_cache(tokens.to(self.model.cfg.device))
        return cache[f"blocks.{self.layer}.mlp.hook_post"][0,:,self.neuron].tolist()

    def text_to_activations_print(self, text):
        token = self.model.to_tokens(text, prepend_bos=False)
        act = self.get_neuron_activation(token)
        act = [f" [{a:.2f}]" for a in act]
        if(token.shape[-1] > 1):
            string = self.model.to_str_tokens(token, prepend_bos=False)
        else: 
            string = self.model.to_string(token)
        res = [None]*(len(string)+len(act))
        res[::2] = string
        res[1::2] = act
        return "".join(res)

    def text_to_visualize(self, text):
        if isinstance(text, str):
            text = [text]
        text_list = []
        act_list = []
        for t in text:
            if isinstance(t, str): # If the text is a list of tokens
                split_text = self.model.to_str_tokens(t, prepend_bos=False)
                token = self.model.to_tokens(t, prepend_bos=False)
            else:
                token = t
                split_text = self.model.to_str_tokens(t, prepend_bos=False)
            text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
            act_list+= self.get_neuron_activation(token) + [0.0]
        act_list = th.tensor(act_list).reshape(-1,1,1)
        return text_neuron_activations(tokens=text_list, activations=act_list)
        # if isinstance(text, list):
        #     text_list = []
        #     act_list = []
        #     for t in text:
        #         split_text = self.model.to_str_tokens(t, prepend_bos=False)
        #         token = self.model.to_tokens(t, prepend_bos=False)
        #         text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
        #         act_list+= self.get_neuron_activation(token) + [0.0]
        #     act_list = th.tensor(act_list).reshape(-1,1,1)
        #     return text_neuron_activations(tokens=text_list, activations=act_list)
        # elif isinstance(text, str):
        #     split_text = self.model.to_str_tokens(text, prepend_bos=False)
        #     token = self.model.to_tokens(text, prepend_bos=False)
        #     act = th.tensor(self.get_neuron_activation(token)).reshape(-1,1,1)
        #     return text_neuron_activations(tokens=split_text, activations=act)
        # else:
        #     raise TypeError("text must be of type str or list, not {type(text)}")
        
    
    def get_text_and_activations_iteratively(self, text):
        tokens = self.model.to_tokens(text, prepend_bos=False)[0]
        original_activation = self.get_neuron_activation(tokens)
        # To get around the newline issue, we replace the newline with \newline and then add a newline at the end
        text_list = [x.replace('\n', '\\newline') for x in self.model.to_str_tokens(text, prepend_bos=False)] + ["\n"]
        act_list = original_activation + [0.0]
        changes = th.zeros(tokens.shape[-1])+100
        for j in range(len(tokens)-1):
            for i in range(len(tokens)):
                changes[i] = self.get_neuron_activation(th.cat((tokens[:i],tokens[i+1:])))[-1]
            max_ind = changes.argmax()
            changes = th.cat((changes[:max_ind], changes[max_ind+1:]))
            tokens = th.cat((tokens[:max_ind],tokens[max_ind+1:]))
            if(tokens.shape[-1] > 1):
                out_text = self.model.to_str_tokens(tokens, prepend_bos=False)
                text_list += [x.replace('\n', '\\newline') for x in out_text] + ["\n"]
            else:
                out_text = self.model.to_string(tokens)
                text_list += [out_text.replace('\n', '\\newline')] + ["\n"]
            act_list += self.get_neuron_activation(tokens) + [0.0]
        text_list = text_list
        act_list = th.tensor(act_list).reshape(-1,1,1)
        return text_list, act_list

    def visualize_text_color_iteratively(self, text):
        if(isinstance(text, str)):
            text_list, act_list = self.get_text_and_activations_iteratively(text)
            return text_neuron_activations(tokens=text_list, activations=act_list)
        elif(isinstance(text, list)):
            text_list_final = []
            act_list_final = []
            for t in range(len(text)):
                text_list, act_list = self.get_text_and_activations_iteratively(text[t])
                text_list_final.append(text_list)
                act_list_final.append(act_list)
            return text_neuron_activations(tokens=text_list_final, activations=act_list_final)

    def simplify_iteratively(self, text):
        # Iteratively remove text that has smallest decrease in activation
        # Print out the change in activation for the largest changes, ie if the change is larger than threshold*original_activation
        tokens = self.model.to_tokens(text, prepend_bos=False)[0]
        self.text_to_activations_print(self.model.to_string(tokens))
        original_activation = self.get_neuron_activation(tokens)[-1]
        changes = th.zeros(tokens.shape[-1])+100
        for j in range(len(tokens)-1):
            for i in range(len(tokens)):
                changes[i] = self.get_neuron_activation(th.cat((tokens[:i],tokens[i+1:])))[-1]
            max_ind = changes.argmax()
            changes = th.cat((changes[:max_ind], changes[max_ind+1:]))
            tokens = th.cat((tokens[:max_ind],tokens[max_ind+1:]))
            out_text = self.model.to_string(tokens)
            print(self.text_to_activations_print(out_text))
        return

    # Assign neuron and layer
    def set_layer_and_neuron(self, layer, neuron):
        self.layer = layer
        self.neuron = neuron
        self.set_hooks()

    def embedded_forward(self, embedded_x):
        self.model_no_embed(embedded_x)
        return self._neurons

    def forward(self, x):
        self.model(x)       
        return self._neurons

    def prompt_optimization(
            self,
            diverse_outputs_num=10, 
            iteration_cap_until_convergence = 30,
            init_text = None,
            seq_size = 4,
            insert_words_and_pos = None, #List of words and positions to insert [word, pos]
            neuron_loss_scalar = 1,
            diversity_loss_scalar = 1,
        ):
        _, _, embed_size = self.model.W_out.shape
        vocab_size = self.model.W_E.shape[0]
        largest_prompts = [None]*diverse_outputs_num
        # Use dim-1 when we're doing a for loop (list comprehension)
        # Use dim-2 when we're doing all at once
        cos_dim_1 = th.nn.CosineSimilarity(dim=1)
        cos_dim_2 = th.nn.CosineSimilarity(dim=2)
        total_iterations = 0

        if init_text is not None:
            init_tokens = self.model.to_tokens(init_text, prepend_bos=False)
            seq_size = init_tokens.shape[-1]
        diverse_outputs = th.zeros(diverse_outputs_num, seq_size, embed_size).to(self.device)
        for d_ind in range(diverse_outputs_num):
            print(f"Starting diverse output {d_ind}")
            if init_text is None:
                # Random tokens of sequence length
                init_tokens = th.randint(0, vocab_size, (1,seq_size))
                init_text = self.model.to_string(init_tokens)
            prompt_embeds = th.nn.Parameter(self.model.embed(init_tokens)).detach()
            prompt_embeds.requires_grad_(True).to(self.device)

            optim = th.optim.AdamW([prompt_embeds], lr=.8, weight_decay=0.01)
            largest_activation = 0
            largest_prompt = None

            iterations_since_last_improvement = 0
            while(iterations_since_last_improvement < iteration_cap_until_convergence):
            # First, project into the embedding matrix
                with th.no_grad():
                    projected_index = th.stack([cos_dim_1(self.embed_weights,prompt_embeds[0,i,:]).argmax() for i in range(seq_size)]).unsqueeze(0)
                    projected_embeds = self.model.embed(projected_index)

                # Create a temp embedding that is detached from the graph, but has the same data as the projected embedding
                tmp_embeds = prompt_embeds.detach().clone()
                tmp_embeds.data = projected_embeds.data
                # add some gaussian noise to tmp_embeds
                # tmp_embeds.data += th.randn_like(tmp_embeds.data)*0.01
                tmp_embeds.requires_grad_(True)

                if insert_words_and_pos is not None:
                    text = insert_words_and_pos[0]
                    pos = insert_words_and_pos[1]
                    if(pos == -1):
                        pos = seq_size
                    token = self.model.to_tokens(text, prepend_bos=False)
                    token_embeds = self.model.embed(token)
                    token_pos = pos
                    wrapped_embeds = th.cat([tmp_embeds[0,:token_pos], token_embeds[0], tmp_embeds[0,token_pos:]], dim=0).unsqueeze(0)
                    if(total_iterations == 0):
                        wrapped_embeds_seq_len = wrapped_embeds.shape[1]
                        projected_index = th.stack([cos_dim_1(self.embed_weights,wrapped_embeds[0,i,:]).argmax() for i in range(wrapped_embeds_seq_len)]).unsqueeze(0)
                        print(f"Inserting {text} at pos {pos}: {self.model.to_str_tokens(projected_index, prepend_bos=False)}")
                else:
                    wrapped_embeds = tmp_embeds

                # Then, calculate neuron_output
                neuron_output = self.embedded_forward(wrapped_embeds)[0,:, self.neuron]
                if(d_ind > 0):
                    diversity_loss = cos_dim_2(tmp_embeds[0], diverse_outputs[:d_ind])
                    # return cos, tmp_embeds, diverse_outputs
                else:
                    diversity_loss = th.zeros(1)

                loss = neuron_loss_scalar*-neuron_output[-1] + diversity_loss_scalar*diversity_loss.mean()

                # Save the highest activation
                if neuron_output[-1] > largest_activation:
                    iterations_since_last_improvement = 0
                    largest_activation = neuron_output[-1]
                    wrapped_embeds_seq_len = wrapped_embeds.shape[1]
                    projected_index = th.stack([cos_dim_1(self.embed_weights,wrapped_embeds[0,i,:]).argmax() for i in range(wrapped_embeds_seq_len)]).unsqueeze(0)
                    largest_prompt = self.model.to_string(projected_index)
                    largest_prompts[d_ind] = largest_prompt
                    print(f"New largest activation: {largest_activation} | {largest_prompt}")

                # Transfer the gradient to the continuous embedding space
                prompt_embeds.grad, = th.autograd.grad(loss, [tmp_embeds])
                
                optim.step()
                optim.zero_grad()
                total_iterations += 1
                iterations_since_last_improvement += 1
                init_text = None
            diverse_outputs[d_ind] = tmp_embeds.data[0,...]
        return largest_prompts
