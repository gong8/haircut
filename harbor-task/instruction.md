Use the Haircut MCP tools to complete one episode. Call `haircut_get_spec` and `haircut_observe` first. Then control the arm and orbit camera with `haircut_step` until the episode ends. Use the two RGB views and the reported joint positions to trim the hair toward its target length.

Aim for zero nicks. Check clearance along the entire move, including the arm links. One nick costs as much as three perfect cells. Back away or move the camera when the view is unclear. Cutting too short is irreversible. End the episode when you have completed the haircut or cannot make further safe progress.

The blade starts parked above the head. Begin with a small approach move toward a nearby cell or re-aim the camera for a clearer view. A cut from the parked position may leave the hair unchanged. If your first cut makes no visible progress, use the returned observations to adjust your position before deciding whether to stop.
