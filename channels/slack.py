from managed_deepagents import channels

channel = channels.slack(
    name="Model Router Agent",
    description="Routes each request to the cheapest model that can handle it",
)
