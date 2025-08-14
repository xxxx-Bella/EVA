import torch

class CoresetSelection(object):
    @staticmethod
    def score_monotonic_selection(data_score, key, ratio, descending, class_balanced):
        """
        data_score: dict with
            - data_score[key]: (N,) tensor of scores
            - data_score['targets']: (N,) tensor of class ids (int)
        key: str, the score key
        ratio: float in (0,1], fraction of data to select
        descending: bool, True -> pick large scores first
        class_balanced: bool, True -> preserve class proportion in selection
        """
        score = data_score[key]
        # global sort indices by score
        score_sorted_index = score.argsort(descending=descending) # high to low
        total_num = ratio * data_score['targets'].shape[0]  # num to be selected. data_score['targets'] is the class label vector for each sample

        if not class_balanced:  # 直接取全局 Top-K
            sel = score_sorted_index[:int(total_num)]
            hi_show = min(15, sel.shape[0])
            print(f'High priority {key}: {score[sel[:hi_show]]}')
            print(f'Low priority {key}: {score[sel[-hi_show:]]}')
            return sel
    
        # ---- class-balanced branch ----
        print('Class balance mode.')
        targets_sorted = data_score['targets'][score_sorted_index]  # sort targets by sorted scores
        classes = torch.unique(targets_sorted)  # category set
        classes, _ = torch.sort(classes)
        
        # per-class indices (positions in the sorted list)
        per_class_positions = {}
        per_class_counts = {}
        for c in classes:
            mask = (targets_sorted == c)
            pos = torch.nonzero(mask, as_tuple=False).squeeze(1)  # positions in sorted order
            per_class_positions[int(c.item())] = pos
            per_class_counts[int(c.item())] = int(mask.sum().item())
            
        # initial quota by proportional allocation
        # q_c = floor(count_c * ratio)
        quotas = {}
        fracs = []  # (fractional_part, class_id) for remainder allocation
        sum_quota = 0
        for cid, cnt in per_class_counts.items():
            raw = cnt * ratio
            q = int(raw)  # floor
            quotas[cid] = q
            sum_quota += q
            fracs.append((raw - q, cid))

        # guarantee at least 1 per class if possible
        # (only for classes with cnt>0; and do not exceed total_num)
        need_min1 = []
        for cid, cnt in per_class_counts.items():
            if cnt > 0 and quotas[cid] == 0:
                need_min1.append(cid)
        # how many slots left to assign
        remain = total_num - sum_quota
        for cid in need_min1:
            if remain <= 0:
                break
            quotas[cid] += 1
            remain -= 1
        
        # If still have remainder, distribute by largest fractional parts
        if remain > 0:
            fracs.sort(reverse=True)  # larger fractional part first
            for _, cid in fracs:
                if remain <= 0:
                    break
                # also cap by available samples
                if quotas[cid] < per_class_counts[cid]:
                    quotas[cid] += 1
                    remain -= 1
        # Final safety: if over-allocated due to corner cases, trim later
        selected_positions = []
        for cid in classes.tolist():
            cid = int(cid)
            pos = per_class_positions[cid]
            k = min(quotas.get(cid, 0), pos.numel())
            if k > 0:
                selected_positions.append(pos[:k])
        
        if len(selected_positions) == 0:
            # fallback: at least pick top total_num globally
            sel_sorted_pos = torch.arange(min(total_num, data_score['targets'].shape[0]))
        else:
            sel_sorted_pos = torch.cat(selected_positions, dim=0)
            # 若合并后超过 total_num（极端边界），截断为前 total_num（仍按全局排序）
            if sel_sorted_pos.numel() > total_num:
                # sel_sorted_pos 本身已按各类内部的“全局排序顺序”，
                # 为严格按全局分数截断，可再整体按位置排序：
                sel_sorted_pos, _ = torch.sort(sel_sorted_pos)
                sel_sorted_pos = sel_sorted_pos[:total_num]

        # Map back to original indices
        selected_index = score_sorted_index[sel_sorted_pos]

        hi_show = min(15, selected_index.shape[0])
        print(f'High priority {key}: {score[selected_index][:hi_show]}')
        print(f'Low  priority {key}: {score[selected_index][-hi_show:]}')

        return selected_index


    @staticmethod
    def mislabel_mask(data_score, mis_key, mis_num, mis_descending, coreset_key):
        mis_score = data_score[mis_key]
        mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
        hard_index = mis_score_sorted_index[:mis_num]
        print(f'Bad data -> High priority {mis_key}: {data_score[mis_key][hard_index][:15]}')
        print(f'Prune {hard_index.shape[0]} samples.')

        easy_index = mis_score_sorted_index[mis_num:]
        data_score[coreset_key] = data_score[coreset_key][easy_index]

        return data_score, easy_index


    @staticmethod
    def stratified_sampling(data_score, coreset_key, coreset_num):
        stratas = 50
        print('Using stratified sampling...')
        score = data_score[coreset_key]
        total_num = coreset_num

        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)

        strata_num = torch.tensor(strata_num)

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]

            num_bin = bins.shape[0]

            rest_exp_num = num
            budgets = []
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                budgets.append(cur_num)
                rest_exp_num -= cur_num


            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)

            return rst

        budgets = bin_allocate(total_num, strata_num)

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(data_score[coreset_key].shape[0])

        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            rand_index = torch.randperm(pool.shape[0])
            selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]

        return selected_index, None

    @staticmethod
    def random_selection(total_num, num):
        print('Random selection.')
        score_random_index = torch.randperm(total_num)

        return score_random_index[:int(num)]
