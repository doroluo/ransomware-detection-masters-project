"""Fixed, label-independent API evidence taxonomy. No strings, paths or architecture features."""
import re

import numpy as np

# Exact normalized API membership, never substring matching ('read' in a name is insufficient).
API_GROUPS = {
    'file_enumerate': 'FindFirstFile FindFirstFileEx FindNextFile NtQueryDirectoryFile GetLogicalDrives GetLogicalDriveStrings',
    'file_open': 'CreateFile CreateFile2 OpenFile NtCreateFile NtOpenFile',
    'file_read': 'ReadFile ReadFileEx NtReadFile ReadFileScatter',
    'file_write': 'WriteFile WriteFileEx NtWriteFile WriteFileGather FlushFileBuffers',
    'file_mutate': 'DeleteFile MoveFile MoveFileEx ReplaceFile SetFileInformationByHandle NtSetInformationFile SetEndOfFile',
    'crypto_transform': 'CryptEncrypt CryptDecrypt BCryptEncrypt BCryptDecrypt CryptHashData BCryptHashData BCryptHash',
    'crypto_key': 'CryptAcquireContext CryptGenKey CryptDeriveKey CryptImportKey CryptExportKey CryptGenRandom BCryptGenRandom BCryptGenerateSymmetricKey BCryptGenerateKeyPair BCryptDeriveKey BCryptDeriveKeyPBKDF2 BCryptImportKey BCryptImportKeyPair NCryptCreatePersistedKey',
    'registry': 'RegOpenKey RegOpenKeyEx RegCreateKey RegCreateKeyEx RegSetValue RegSetValueEx RegDeleteKey RegDeleteValue RegQueryValueEx',
    'process': 'CreateProcess CreateProcessAsUser OpenProcess TerminateProcess ShellExecute ShellExecuteEx WinExec',
    'service': 'OpenSCManager OpenService CreateService StartService ControlService DeleteService ChangeServiceConfig',
    'network': 'connect send recv sendto recvfrom WSASend WSARecv InternetOpen InternetConnect InternetOpenUrl HttpOpenRequest HttpSendRequest WinHttpOpen WinHttpConnect WinHttpOpenRequest WinHttpSendRequest URLDownloadToFile',
}
TAG_NAMES = tuple(API_GROUPS)
API_TO_TAG = {api.lower(): i for i, names in enumerate(API_GROUPS.values()) for api in names.split()}


def api_tag(symbol):
    name = symbol.split('!', 1)[-1].lower()
    name = re.sub(r'@\d+$', '', name)
    name = re.sub(r'^(?:__imp_)?_?', '', name)
    if name not in API_TO_TAG and name.endswith(('a', 'w')):
        name = name[:-1]
    return API_TO_TAG.get(name)


def function_tags(z, api_vocab):
    """Binary direct-import-reference tags and one-hop callee tags, aligned to fn_flags.

    Uses precisely the cache's import references (including references that are not
    direct calls). A tag is static API association, never proof of runtime behavior.
    Callee tags exclude the function's own evidence; pass both matrices separately
    if desired. No fixed-point propagation that could flood a large call component.
    """
    tags = np.zeros((len(z['fn_flags']), len(TAG_NAMES)), dtype=np.float32)
    lut = np.array([api_tag(s) if api_tag(s) is not None else -1 for s in api_vocab], dtype=np.int32)
    ptr = z['blk_api_ptr'][z['fn_blk_ptr']]
    for fn, (lo, hi) in enumerate(zip(ptr[:-1], ptr[1:])):
        ids = z['api'][lo:hi]
        if len(ids) and (ids.min() < 0 or ids.max() >= len(lut)):
            raise ValueError('API IDs outside vocabulary')
        indices = lut[ids]
        tags[fn, indices[indices >= 0]] = 1
    callee = np.zeros_like(tags)
    for src, dst in z['call_edges']:
        callee[src] = np.maximum(callee[src], tags[dst])
    return tags, callee
