import torch

def custom_meshgrid(*args):
    return torch.meshgrid(*args, indexing="ij")


def get_relative_pose_from_c2w_torch(c2ws: torch.Tensor) -> torch.Tensor:
    """
    c2ws: [T, 4, 4]
    return: [T, 4, 4], relative to first frame
    """
    first_w2c = torch.linalg.inv(c2ws[0])
    rel = torch.matmul(first_w2c.unsqueeze(0), c2ws)  # [T,4,4]
    rel[0] = torch.eye(4, device=c2ws.device, dtype=c2ws.dtype)
    return rel


def ray_condition_from_K_c2w(Ks_4: torch.Tensor, c2ws: torch.Tensor, H: int, W: int, device):
    """
    Ks_4: [1, T, 4]  -> fx, fy, cx, cy
    c2ws: [1, T, 4, 4]
    return: [1, T, H, W, 6]
    """
    B = Ks_4.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2ws.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2ws.dtype),
    )
    i = i.reshape(1, 1, H * W).expand(B, 1, H * W) + 0.5
    j = j.reshape(1, 1, H * W).expand(B, 1, H * W) + 0.5

    fx, fy, cx, cy = Ks_4.chunk(4, dim=-1)  # [1, T, 1]

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)   # [1, T, HW, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2ws[..., :3, :3].transpose(-1, -2)  # [1, T, HW, 3]
    rays_o = c2ws[..., :3, 3]                                  # [1, T, 3]
    rays_o = rays_o[:, :, None, :].expand_as(rays_d)           # [1, T, HW, 3]

    rays_dxo = torch.linalg.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)            # [1, T, HW, 6]
    plucker = plucker.reshape(B, c2ws.shape[1], H, W, 6)       # [1, T, H, W, 6]
    return plucker


def process_known_camera_tensors(vace_camera, width, height, device="cpu", use_relative_pose=True):
    """
    vace_camera:
    {
        "Ks": [T,3,3],
        "c2ws": [T,4,4]   # 这里实际存的是 c2w
    }
    return:
        [T, H, W, 6]
    """
    Ks = vace_camera["Ks"].to(device=device, dtype=torch.float32)          # [T,3,3]
    c2ws = vace_camera["c2ws"].to(device=device, dtype=torch.float32)  # [T,4,4]

    if use_relative_pose:
        c2ws = get_relative_pose_from_c2w_torch(c2ws)

    fx = Ks[:, 0, 0]
    fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2]
    cy = Ks[:, 1, 2]

    Ks_4 = torch.stack([fx, fy, cx, cy], dim=-1)[None]   # [1,T,4]
    c2ws = c2ws[None]                                    # [1,T,4,4]

    plucker = ray_condition_from_K_c2w(Ks_4, c2ws, height, width, device=device)[0]  # [T,H,W,6]
    return plucker
