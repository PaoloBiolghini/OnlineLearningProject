class Auction:
    def __init__(self, *args, **kwargs):
        pass

    def get_winners(self, bids):
        pass

    def get_payments_per_click(self, winners, values, bids):
        pass

    def round(self, bids):
        winners, values = self.get_winners(bids) 
        payments_per_click = self.get_payments_per_click(winners, values, bids)
        return winners, payments_per_click